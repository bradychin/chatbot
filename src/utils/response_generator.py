# --------- Import Libraries ---------#
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import sys
import re
from src.log.logger import get_logger
logger = get_logger(__name__)

# --------- Functions ---------#
def load_model(saved_model): # Make function
    """Load model and tokenizer"""
    try:
        model = GPT2LMHeadModel.from_pretrained(saved_model)
        tokenizer = GPT2Tokenizer.from_pretrained(saved_model)
        logger.info('Model and Tokenizer load success.')
    except Exception as e:
        logger.error(f'Failed to load Model or Tokenizer: {e}')
        sys.exit(1)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    return model, tokenizer, device

def clean_response(response): # Make function
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

# --------- Response Generator ---------#
class ResponseGenerator:
    def __init__(self, model_path):
        self.model, self.tokenizer, self.device = load_model(model_path)

    def generate(self, prompt):
        """Generate response for the given prompt."""
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

        response_text = clean_response(response_text)

        return response_text.capitalize()

