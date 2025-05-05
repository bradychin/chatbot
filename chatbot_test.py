# --------- Import libraries ---------#
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------- Response Generator ---------#
class ResponseGenerator:
    def __init__(self, model_path):
        self.model, self.tokenizer, self.device = self._load_model(model_path)

    def _load_model(self, model_path):
        """Load the model and tokenizer from the specified path."""
        try:
            model = GPT2LMHeadModel.from_pretrained(model_path)
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            logger.info('Model and Tokenizer loaded successfully.')
        except Exception as e:
            logger.error(f'Failed to load model or tokenizer: {e}')
            raise

        # Configure tokenizer padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        return model, tokenizer, device

    def generate(self, prompt):
        """Generate a response for the given prompt."""
        # Format and encode prompt
        formatted_prompt = f'{self.tokenizer.bos_token}{prompt}'
        encoded_prompt = self.tokenizer.encode(formatted_prompt, return_tensors='pt').to(self.device)

        # Generate response
        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            max_new_tokens=150,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            repetition_penalty=1.1
        )

        # Decode and extract response
        complete_response = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        input_text = self.tokenizer.decode(encoded_prompt[0], skip_special_tokens=True)
        response_text = complete_response[len(input_text):]

        # Clean up the response
        if self.tokenizer.eos_token in response_text:
            response_text = response_text[:response_text.find(self.tokenizer.eos_token)]

        response_text = self._clean_response(response_text)
        return response_text.capitalize()

    def _clean_response(self, response):
        """Clean and format the response text."""
        # Basic cleaning
        response = re.sub(r'\s+', ' ', response).strip()

        # Remove unwanted prefixes
        response = re.sub(r'(?i)(^|\s)(user:|bot:)', '', response)

        # Remove non-standard characters
        response = re.sub(r'[^\w\s.,?!\'"-]', '', response)

        # Standardize quotes
        response = re.sub(r'[""]', '"', response)
        response = re.sub(r"['']", "'", response)

        # Remove user/bot references
        response = re.sub(r'(?i)\b(user|bot)\b', '', response)

        # Fix spacing around punctuation
        response = re.sub(r'\s+([.,!?;:])', r'\1', response)

        # Clean up extra spaces
        response = re.sub(r'\s+', ' ', response).strip()

        # Take only first line
        if '\n' in response:
            response = response.split('\n')[0]

        return response


# --------- Chatbot ---------#
class Chatbot:
    def __init__(self, response_generator):
        self.response_generator = response_generator
        self.session_context = 'You are a helpful chatbot.'

    def generate_response(self, user_input):
        """Generate a response to the user input."""
        if not user_input.strip():
            return 'I did not receive an input'

        prompt = f'{self.session_context}\nUser: {user_input}\nBot:'
        response = self.response_generator.generate(prompt)

        # Fallback for very short or invalid responses
        if len(response.split()) < 3 or not any(char.isalpha() for char in response):
            response = 'I need to think about that. Could you ask me something else?'

        return response


# --------- Main ---------#
def main():
    """Run the chatbot application."""
    model_path = 'models/model_rev2'  # Path to your trained model

    try:
        response_generator = ResponseGenerator(model_path)
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
                logger.error(f"Error occurred: {e}")
                print("An error occurred. Let's continue our conversation.")
                continue

    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        print("Could not start the chatbot. Please check the logs for details.")


if __name__ == '__main__':
    main()