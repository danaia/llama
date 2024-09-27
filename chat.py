import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os
import re

# Limit the number of threads PyTorch uses
torch.set_num_threads(4)  # Adjust based on your CPU cores
print(f"Using {torch.get_num_threads()} threads for PyTorch.")

# Define the apply_chat_template function
def apply_chat_template(messages, tokenizer, return_tensors="pt"):
    if not messages:
        return None  # Handle empty messages gracefully

    # Extract the last user message
    last_user_message = messages[-1]['content']
    # Construct the prompt using "Q:" and "A:" format
    prompt = f"Q: {last_user_message}\nA:"

    print(f"Constructed prompt: {prompt}")  # Verbose logging

    return tokenizer(
        prompt, 
        return_tensors=return_tensors, 
        truncation=True, 
        max_length=512,  # Adjust as needed
        padding=True
    )

# Function to clean the model's response
def clean_response(response):
    # Remove numbered lists (e.g., "1. ")
    response = re.sub(r'^\d+\.\s*', '', response, flags=re.MULTILINE)
    
    # Remove trailing incomplete sentences or asterisks
    response = re.sub(r'\*.*$', '', response, flags=re.DOTALL)
    
    # Optionally, remove any residual special characters except basic punctuation
    response = re.sub(r'[^\w\s\.,!?]', '', response)
    
    return response.strip()

# Function to load model and tokenizer without dynamic quantization
def load_model_and_tokenizer():
    model_path = "/home/art/dev/llama/llama-3.2-1B"  # Update if using a different model
    print(f"Attempting to load model from: {model_path}")
    
    if not os.path.exists(model_path) and not os.path.exists(os.path.join(model_path, "config.json")):
        raise ValueError(f"Model path does not exist or is invalid: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    print("Fast tokenizer loaded successfully")

    # Handle multiple eos_token_ids by selecting the first one
    if isinstance(tokenizer.eos_token_id, list):
        tokenizer.eos_token_id = tokenizer.eos_token_id[0]
    elif tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.pad_token_id
    print(f"eos_token_id: {tokenizer.eos_token_id}")

    # Ensure pad_token_id is set to the first eos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"pad_token_id was not set. Setting pad_token_id to eos_token_id: {tokenizer.pad_token_id}")
    else:
        print(f"pad_token_id: {tokenizer.pad_token_id}")

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="cpu",
        low_cpu_mem_usage=True,
        pad_token_id=tokenizer.pad_token_id
    )
    print("Model loaded successfully")

    # Optionally apply dynamic quantization (commented out for testing)
    # print("Applying dynamic quantization to the model...")
    # model = torch.quantization.quantize_dynamic(
    #     model, 
    #     {torch.nn.Linear},  # Specify layers to quantize
    #     dtype=torch.qint8
    # )
    # print("Dynamic quantization applied successfully")

    # Check model configuration
    config = AutoConfig.from_pretrained(model_path)
    print(f"Model configuration:\n{config}")

    return tokenizer, model

# Function to generate response with error handling and attention mask
def generate_response(messages, tokenizer, model):
    try:
        inputs = apply_chat_template(messages, tokenizer, return_tensors="pt")
        if inputs is None:
            return "I'm here to help! How can I assist you today?"

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        print("Starting generation...")  # Verbose logging

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,        # Adjust for desired response length
                do_sample=True,            # Enables sampling for diversity
                temperature=0.6,           # Lowered for more coherent responses
                top_p=0.9,                 # Nucleus sampling
                top_k=50,                  # Limits sampling to top 50 tokens
                repetition_penalty=1.1,    # Slightly discourages repetition
                no_repeat_ngram_size=3,    # Prevents repeating trigrams
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id
            )
        
        print("Generation completed.")  # Verbose logging
        
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = decoded_output.strip()
        response = clean_response(response)
        
        print(f"Generated response: {response}")  # Verbose logging
        
        return response
    except Exception as e:
        print(f"An error occurred during generation: {e}")
        return "I'm sorry, I encountered an error while generating a response."

# Main chat loop with history management
def main():
    print("Loading model and tokenizer...")
    try:
        tokenizer, model = load_model_and_tokenizer()
        model.eval()  # Set model to evaluation mode
        print("Model is in evaluation mode.")
    except Exception as e:
        print(f"Error loading model and tokenizer: {e}")
        return

    # Initialize messages with an initial instruction
    messages = [
        {
            "role": "system",
            "content": "You are a friendly and knowledgeable assistant. Provide clear and concise answers to the user's questions."
        }
    ]

    max_history = 5  # Adjust based on memory constraints

    print("Welcome to the Llama-3.2-1B chat! Type 'quit' to exit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        messages.append({"role": "user", "content": user_input})
        print("Generating response...")  # Verbose logging

        response = generate_response(messages, tokenizer, model)
        messages.append({"role": "assistant", "content": response})

        # Trim history to maintain only recent exchanges
        if len(messages) > max_history + 1:
            messages = messages[:1] + messages[-max_history:]

        print("Assistant:", response)

    print("Thank you for chatting!")

if __name__ == "__main__":
    main()
