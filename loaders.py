import sys
import logging

logger = logging.getLogger('chatbot')

def safe_load(loader_fn, name):
    try:
        return loader_fn()
    except (OSError, FileNotFoundError) as e:
        logger.error(f'{name} load failed: {e}')
        sys.exit(1)

# Example usage:
# from loaders import safe_load
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
#
# model = safe_load(lambda: GPT2LMHeadModel.from_pretrained(saved_model), 'Model')
# tokenizer = safe_load(lambda: GPT2Tokenizer.from_pretrained(saved_model), 'Tokenizer')