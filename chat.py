import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the apply_chat_template function
def apply_chat_template(messages, return_tensors="pt"):
    # Combine the messages into a single prompt string
    prompt = ""
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        prompt += f"{role.capitalize()}: {content}\n"
    # Add the assistant prompt
    prompt += "Assistant:"
    
    # Tokenize the prompt
    return tokenizer(prompt, return_tensors=return_tensors)

# Load the tokenizer and model
model_id = "/Users/dana/dev/llamaChat/llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Ensure the tokenizer has an eos_token_id
if tokenizer.eos_token_id is None:
    tokenizer.eos_token_id = tokenizer.pad_token_id

# Load the model with pad_token_id set to eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_length=64,
    pad_token_id=tokenizer.eos_token_id  # Explicitly set pad_token_id
)

# Function to generate response with error handling and attention mask
def generate_response(messages):
    try:
        # Prepare the input by applying the chat template
        inputs = apply_chat_template(messages, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Attention Mask shape: {attention_mask.shape}")
        
        # Generate outputs with attention_mask
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7
        )
        
        print(f"Outputs shape: {outputs.shape}")
        print(f"Outputs tensor: {outputs}")
        
        # Decode the generated outputs
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        print(f"Decoded Outputs: {decoded_outputs}")
        
        # Assuming batch_size=1, extract the first (and only) response
        response = decoded_outputs[0].split("Assistant:")[-1].strip()
        
        return response
    except Exception as e:
        print(f"An error occurred during generation: {e}")
        return "I'm sorry, I encountered an error while generating a response."

# Main chat loop with history management
def main():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    max_history = 5  # Adjust based on your requirements

    print("Welcome to the Llama-3.2-1B-Instruct chat! Type 'quit' to exit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        messages.append({"role": "user", "content": user_input})
        response = generate_response(messages)
        messages.append({"role": "assistant", "content": response})

        # Trim the history to maintain only the recent exchanges
        if len(messages) > max_history + 1:
            messages = messages[:1] + messages[-max_history:]

        print("Assistant:", response)

    print("Thank you for chatting!")

if __name__ == "__main__":
    main()
