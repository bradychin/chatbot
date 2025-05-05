from torch.utils.data import Dataset

from src import config

class DialogManager(Dataset):
    """Dialog dataset management for training."""
    def __init__(self, dialog, tokenizer, max_length=config.max_length):
        self.dialog = dialog
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialog)

    def __getitem__(self, idx):
        """Process conversations."""
        messages = self.dialog[idx]

        combined_text = f'{self.tokenizer.bos_token}'

        for i in range(len(messages) - 1):
            combined_text = f'{messages[i]}'

            if i < len(messages) - 1:
                combined_text += '\n'

        combined_text += f'{self.tokenizer.eos_token}'

        encodings = self.tokenizer(combined_text,
                                   truncation=True,
                                   max_length=self.max_length,
                                   padding='max_length',
                                   return_tensors='pt')

        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()

        # For GPT training, labels are the same as inputs
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids
        }
