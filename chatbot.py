#--------- Import libraries ---------#
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import sys
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#--------- Response Generator ---------#
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
        formatted_prompt = f'{self.tokenizer.bos_token} {prompt} {self.tokenizer.eos_token}'
        encoded_prompt = self.tokenizer.encode(formatted_prompt, return_tensors='pt').to(self.device)

        output_sequences = self.model.generate(input_ids=encoded_prompt,
                                          max_length=100,
                                          temperature=0.8,
                                          top_k=40,
                                          top_p=0.9,
                                          do_sample=True,
                                          num_return_sequences=1,
                                          pad_token_id=self.tokenizer.pad_token_id,
                                          eos_token_id=self.tokenizer.eos_token_id)

        complete_response = self.tokenizer.decode(output_sequences[0], skip_special_tokens=False)
        input_text = self.tokenizer.decode(encoded_prompt[0], skip_special_tokens=False)
        response_text = complete_response[len(input_text):]

        if self.tokenizer.eos_token in response_text:
            response_text = response_text[:response_text.find(self.tokenizer.eos_token)]

        if prompt == 'stop':
            response_text = 'Have a good day!'

        response_text = re.sub(r'\s+', ' ', response_text).strip()
        response_text = re.sub(r'[^\w\s.,?!\'"-]', '', response_text)
        response_text = re.sub(r'[""]', '"', response_text)
        response_text = re.sub(r"['']", "'", response_text)

        return response_text.capitalize()

#--------- Chatbot ---------#
class Chatbot:
    pass

#--------- Main ---------#
def main():
    saved_model = 'Models/chatbot_model'
    response_generator = ResponseGenerator(saved_model)

    print('\nIf you want to stop the conversation you can type "stop".\n')
    while True:
        prompt = input('>>> ')
        response = response_generator.generate(prompt)

        print(f'Chatbot: {response}\n')

        if prompt == 'stop':
            break

if __name__ == '__main__':
    main()