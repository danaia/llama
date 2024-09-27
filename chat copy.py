import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def apply_chat_template(messages, tokenizer, return_tensors="pt"):
    prompt = ""
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        prompt += f"{role.capitalize()}: {content}\n"
    prompt += "Assistant:"
    return tokenizer(prompt, return_tensors=return_tensors, truncation=True, max_length=256)

def load_model_and_tokenizer():
    model_path = os.path.expanduser("~/dev/llamaChat/llama-3.2-1B-Instruct")
    print(f"Attempting to load model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Tokenizer loaded successfully")

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.pad_token_id
    print(f"eos_token_id: {tokenizer.eos_token_id}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="cpu",
        low_cpu_mem_usage=True,
        pad_token_id=tokenizer.eos_token_id
    )
    print("Model loaded successfully")

    return tokenizer, model

def generate_response(messages, tokenizer, model):
    try:
        inputs = apply_chat_template(messages, tokenizer, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = decoded_output.split("Assistant:")[-1].strip()
        
        return response
    except Exception as e:
        print(f"An error occurred during generation: {e}")
        return "I'm sorry, I encountered an error while generating a response."

def main():
    print("Loading model and tokenizer...")
    try:
        tokenizer, model = load_model_and_tokenizer()
    except Exception as e:
        print(f"Error loading model and tokenizer: {e}")
        return

    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    max_history = 3  # Reduced for memory efficiency

    print("Welcome to the Llama-3.2-1B-Instruct chat! Type 'quit' to exit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        messages.append({"role": "user", "content": user_input})
        response = generate_response(messages, tokenizer, model)
        messages.append({"role": "assistant", "content": response})

        if len(messages) > max_history + 1:
            messages = messages[:1] + messages[-max_history:]

        print("Assistant:", response)

    print("Thank you for chatting!")

if __name__ == "__main__":
    main()