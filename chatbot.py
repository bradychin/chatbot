# --------- Import libraries ---------#
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import sys
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------- Response Generator ---------#
class ResponseGenerator:
    def __init__(self, saved_model):
        self.model, self.tokenizer, self.device = self.load_model(saved_model)

    def load_model(self, saved_model):
        try:
            model = GPT2LMHeadModel.from_pretrained(saved_model)
            logger.info('Model load success.')
        except OSError:
            logger.error('Model load was not successful.')
            sys.exit(1)
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(saved_model)
            logger.info('Tokenizer load success.')
        except OSError:
            logger.error('Tokenizer load was not successful.')
            sys.exit(1)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        return model, tokenizer, device

    def generate(self, prompt):
        formatted_prompt = f'{self.tokenizer.bos_token}{prompt}'
        encoded_prompt = self.tokenizer.encode(formatted_prompt, return_tensors='pt').to(self.device)

        output_sequences = self.model.generate(input_ids=encoded_prompt,
                                               max_new_tokens=100,
                                               temperature=1.0,
                                               top_k=50,
                                               top_p=0.92,
                                               do_sample=True,
                                               num_return_sequences=1,
                                               pad_token_id=self.tokenizer.pad_token_id,
                                               eos_token_id=self.tokenizer.eos_token_id,
                                               no_repeat_ngram_size=3)

        complete_response = self.tokenizer.decode(output_sequences[0], skip_special_tokens=False)
        input_text = self.tokenizer.decode(encoded_prompt[0], skip_special_tokens=False)
        response_text = complete_response[len(input_text):]

        if self.tokenizer.eos_token in response_text:
            response_text = response_text[:response_text.find(self.tokenizer.eos_token)]

        response_text = re.sub(r'\s+', ' ', response_text).strip()
        response_text = re.sub(r'[^\w\s.,?!\'"-]', '', response_text)
        response_text = re.sub(r'[""]', '"', response_text)
        response_text = re.sub(r"['']", "'", response_text)

        return response_text.capitalize()

# --------- Chatbot ---------#
class Chatbot:
    def __init__(self, response_generator, max_history=5):
        self.response_generator = response_generator
        self.max_history = max_history
        self.conversation_history = []

    def add_to_history(self, role, message):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": message})
        # Keep only the most recent messages based on max_history
        if len(self.conversation_history) > self.max_history * 2:  # *2 because each turn has 2 messages (user + bot)
            self.conversation_history = self.conversation_history[-self.max_history * 2:]

    def format_conversation_for_prompt(self):
        """Format the conversation history into a prompt for the model."""
        formatted_history = ""
        for entry in self.conversation_history:
            prefix = "User: " if entry["role"] == "user" else "Bot: "
            formatted_history += f"{prefix}{entry['content']}\n"
        return formatted_history.strip()

    def generate_response(self, user_input):
        """Generate a response based on the user input and conversation history."""
        # Add user message to history
        self.add_to_history("user", user_input)

        # Build prompt with conversation context
        if len(self.conversation_history) <= 2:  # First exchange
            prompt = f"User: {user_input}\nBot:"
        else:
            # Create a context window from recent conversation
            # Format the conversation in a way that helps the model better understand the flow
            prompt = self.format_conversation_for_prompt()
            prompt += "\nBot:"  # Add prompt for bot to continue

        # Generate response using the conversation-formatted prompt
        response = self.response_generator.generate(prompt)

        # Clean up the response if it starts with "Bot:" (which can happen)
        if response.lower().startswith("bot:"):
            response = response[4:].strip()

        # Add bot response to history
        self.add_to_history("bot", response)

        return response

# --------- Main ---------#
def main():
    saved_model = 'Models/model'
    response_generator = ResponseGenerator(saved_model)
    chatbot = Chatbot(response_generator)

    print('\nWelcome to the chatbot! Type "stop" at any time to end the conversation.\n')

    while True:
        try:
            user_input = input('>>> ')

            if user_input.lower() == "stop":
                print("Chatbot: Have a good day!")
                break

            response = chatbot.generate_response(user_input)
            print(f'Chatbot: {response}\n')

        except Exception as e:
            print(f"Error occurred: {e}")
            print("Restarting conversation...")
            continue

if __name__ == '__main__':
    main()