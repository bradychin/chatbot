#--------- Import libraries ---------#
from transformers import GPT2Tokenizer
import re
import sys
import pandas as pd
from torch.utils.data import DataLoader, random_split
from src.log.logger import get_logger
logger = get_logger(__name__)

#--------- Import Classes ---------#
from src.utils.dialog_manager import DialogManager

#--------- Data Processor ---------#
def preprocess_text(text): # Make function
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s.,?!\'"-]', '', text)
    text = re.sub(r'[""]', '"', text)
    text = re.sub(r"['']", "'", text)
    return text

class DataProcessor:
    def __init__(self, data_file_path, train_val_split=0.2):
        self.data_file_path = data_file_path
        self.train_val_split = train_val_split

        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        special_tokens = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
        self.tokenizer.add_special_tokens(special_tokens)

        try:
            self.dialog_data = self._load_dialog_data()
            logger.info(f'Created {len(self.dialog_data)} dialog paris.')
        except FileNotFoundError:
            logger.error(f'File "{data_file_path}" not found.')
            sys.exit(1)

        self.train_dataset, self.val_dataset = self._create_datasets()

    # Create dialog
    def _load_dialog_data(self):
        logger.info(f'Loading data from {self.data_file_path}')
        df = pd.read_csv(self.data_file_path)
        dialog_data = []

        for _, row in df.iterrows():
            chat_text = row['chat']

            if not isinstance(chat_text, str) or not chat_text.strip():
                continue

            # Split chat into messages
            messages = [m.strip() for m in chat_text.split('\n') if m.strip()]

            if len(messages) < 2:
                continue

            clean_messages = [preprocess_text(message) for message in messages]
            clean_messages = [message for message in clean_messages if len(message.split()) <= 50]

            if len(clean_messages) >= 2:
                for window_size in range(2, min(9, len(clean_messages) + 1)):
                    for start_idx in range(0, len(clean_messages) - window_size + 1):
                        window = clean_messages[start_idx:start_idx + window_size]
                        if all(len(message.split()) >= 3 for message in window):
                            dialog_data.append(window)


        logger.info(f'Created {len(dialog_data)} training examples.')
        return dialog_data

    # Create datasets
    def _create_datasets(self):
        """Split into training and validation sets."""
        full_dataset = DialogManager(self.dialog_data, self.tokenizer)
        validation_size = int(len(full_dataset) * self.train_val_split)
        train_size = len(full_dataset) - validation_size
        train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])

        logger.info(f'Training dataset: {train_size} dialog pairs.')
        logger.info(f'Validation dataset: {validation_size} dialog pairs.\n')

        return train_dataset, validation_dataset

    # Load train and validation data
    def get_dataloaders(self, batch_size=4):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  pin_memory=True)
        validation_loader = DataLoader(self.val_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=2,
                                       pin_memory=True)

        return train_loader, validation_loader