class TextIteratorStreamer:
    def __init__(self, tokenizer, skip_prompt=False, skip_special_tokens=False):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.tokens = []
        self.text_queue = []

    def put(self, value):
        if isinstance(value, torch.Tensor):
            value = value.cpu().tolist()
        
        # Ensure value is always a list
        if not isinstance(value, list):
            value = [value]
        
        self.tokens.extend(value)
        
        # Decode only the new tokens
        new_text = self.tokenizer.decode(value, skip_special_tokens=self.skip_special_tokens)
        if new_text:
            self.text_queue.append(new_text)

    def end(self):
        if self.skip_prompt:
            # Find the last occurrence of "Assistant:" and keep only the text after it
            full_text = self.tokenizer.decode(self.tokens, skip_special_tokens=self.skip_special_tokens)
            assistant_text = full_text.split("Assistant:")[-1].strip()
            if assistant_text:
                self.text_queue.append(assistant_text)
        elif self.tokens:
            remaining_text = self.tokenizer.decode(self.tokens, skip_special_tokens=self.skip_special_tokens)
            if remaining_text:
                self.text_queue.append(remaining_text)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.text_queue) == 0:
            raise StopIteration()
        return self.text_queue.pop(0)