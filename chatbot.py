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
            tokenizer = GPT2Tokenizer.from_pretrained(saved_model)
            logger.info('Model and Tokenizer load success.')
        except:
            logger.error('Model or Tokenizer load was not successful.')
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
                                               max_new_tokens=150,
                                               temperature=0.7,
                                               top_k=50,
                                               top_p=0.95,
                                               do_sample=True,
                                               num_return_sequences=1,
                                               pad_token_id=self.tokenizer.pad_token_id,
                                               eos_token_id=self.tokenizer.eos_token_id,
                                               no_repeat_ngram_size=2,
                                               repetition_penalty=1.1)

        complete_response = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        input_text = self.tokenizer.decode(encoded_prompt[0], skip_special_tokens=True)
        response_text = complete_response[len(input_text):]

        if self.tokenizer.eos_token in response_text:
            response_text = response_text[:response_text.find(self.tokenizer.eos_token)]

        response_text = self.clean_response(response_text)

        return response_text.capitalize()

    def clean_response(self, response):
        # More aggressive cleaning to remove problematic patterns
        response = re.sub(r'\s+', ' ', response).strip()

        # Remove any "user:" or "bot:" prefixes that might have leaked into the output
        response = re.sub(r'(?i)(^|\s)(user:|bot:)', '', response)

        # Remove strange tokens or markers that might appear
        response = re.sub(r'[^\w\s.,?!\'"-]', '', response)

        # Standardize quotes
        response = re.sub(r'[""]', '"', response)
        response = re.sub(r"['']", "'", response)

        # Remove user/bot references
        response = re.sub(r'(?i)\b(user|bot)\b', '', response)

        response = re.sub(r'\s+([.,!?;:])', r'\1', response)

        # Clean up spacing after removing terms
        response = re.sub(r'\s+', ' ', response).strip()

        # Truncate at first newline to avoid multi-turn responses
        if '\n' in response:
            response = response.split('\n')[0]

        return response

# --------- Chatbot ---------#
class Chatbot:
    def __init__(self, response_generator, max_history=8):
        self.response_generator = response_generator
        self.session_context = 'You are a helpful chatbot.'

    def generate_response(self, user_input):
        if not user_input.strip():
            return 'I did not receive an input'

        prompt = f'{self.session_context}\nUser: {user_input}\nBot:'
        response = self.response_generator.generate(prompt)

        if len(response.split()) < 3 or not any(char.isalpha() for char in response):
            response = 'I need to think about that. Could you ask me something else?'

        return response

# --------- Main ---------#
def main():
    # saved_model = 'models/checkpoints/checkpoint_epoch_0_batch_2000_lr2e-5'
    saved_model = 'models/model_rev2'
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