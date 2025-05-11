#--------- Import libraries ---------#
from torch.utils.data import Dataset

#--------- Import Scripts ---------#
from src import config

#--------- Dialog Manager ---------#
class DialogManager(Dataset):
    def __init__(self, dialog_pairs, tokenizer, max_length=config.max_length):
        self.dialog_pairs = dialog_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialog_pairs)

    def __getitem__(self, idx):
        conversation = self.dialog_pairs[idx]

        if isinstance(conversation, list) and len(conversation) > 1:
            combined_text = f'{self.tokenizer.bos_token}'

            for i in range(0, len(conversation) - 1, 2):
                if i + 1 < len(conversation):
                    user_text = conversation[i]
                    response_text = conversation[i + 1]
                    combined_text += f'{user_text}\n{response_text}'

                    if i + 2 < len(conversation):
                        combined_text += '\n'
            combined_text += f'{self.tokenizer.eos_token}'
        else:
            prompt, response = conversation if isinstance(conversation, tuple) else (conversation[0], conversation[1])
            combined_text = f'{self.tokenizer.bos_token}{prompt}\n{response} {self.tokenizer.eos_token}'

        encodings = self.tokenizer(combined_text,
                                   truncation=True,
                                   max_length=self.max_length,
                                   padding="max_length",
                                   return_tensors="pt")

        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()

        # For GPT training, labels are the same as inputs
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids
        }