#--------- 1. Import libraries ---------#
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#--------- Tokenize ---------#
class DialogManagement(Dataset):
    def __init__(self, dialog_pairs, tokenizer, max_length=512):
        self.dialog_pairs = dialog_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialog_pairs)

    def __getitem__(self, idx):
        prompt, response = self.dialog_pairs[idx]
        # Format: <BOS> prompt <EOS> response <EOS>
        combined_text = f"{self.tokenizer.bos_token} {prompt} {self.tokenizer.eos_token} {response} {self.tokenizer.eos_token}"

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

#--------- Process Data ---------#
class ProcessData:
    def __init__(self, movie_lines_path, movie_conversations_path):
        self.movie_lines_path = movie_lines_path
        self.movie_conversations_path = movie_conversations_path
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        special_tokens = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
        self.tokenizer.add_special_tokens(special_tokens)
        self.lines = self.load_lines()
        self.conversations = self.load_conversations()
        self.dialog = self.create_dialog()

    # Import lines
    def load_lines(self):
        lines = {}
        with open(self.movie_lines_path, 'r', encoding='iso-8859-1') as file:
            for line in file:
                parts = line.strip().split(' +++$+++ ')
                line_id, text = parts[0], parts[-1]
                lines[line_id] = text
        return lines

    # Import conversations
    def load_conversations(self):
        conversations = []
        with open(self.movie_conversations_path, 'r', encoding='iso-8859-1') as file:
            for line in file:
                parts = line.strip().split(' +++$+++ ')
                line_ids = eval(parts[-1])
                conversations.append(line_ids)
        return conversations

    # Create dialog
    def create_dialog(self):
        dialog = []
        for conversation in self.conversations:
            for i in range(len(conversation)-1):
                prompt = self.lines[conversation[i]]
                response = self.lines[conversation[i+1]]
                dialog.append((prompt, response))
        return dialog

    def get_dataset(self):
        return DialogManagement(self.dialog, self.tokenizer)

#--------- Chatbot ---------#
class Chatbot:
    def __init__(self, movie_lines_path, movie_conversations_path):
        self.data_processor = ProcessData(movie_lines_path, movie_conversations_path)
        self.tokenizer = self.data_processor.tokenizer
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.resize_token_embeddings(len(self.tokenizer))

    def fine_tune(self):
        dataset = self.data_processor.get_dataset()
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {device}')

        self.model.to(device)
        optimizer = AdamW(self.model.parameters(), lr=5e-6)

        # Fine tune
        self.model.train()
        epochs = 3
        for epoch in range(epochs):
            for batch_index, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                if input_ids.size(0) == 0:
                    print('Empty batch. Skipping')
                    continue
                outputs = self.model(input_ids = input_ids,
                                     attention_mask = attention_mask,
                                     labels = labels)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f'Epoch: {epoch}. Loss: {loss.item()}')

        # self.model.save_pretrained("fine_tuned_chatbot")
        # self.data_processor.tokenizer.save_pretrained("fine_tuned_chatbot")

#--------- Main Function ---------#
def main():
    movie_lines_path = 'test_lines.txt'
    # movie_lines_path = 'cornell movie-dialogs corpus/movie_lines.txt'
    movie_conversations_path = 'test_conversations.txt'
    # movie_conversations_path = 'cornell movie-dialogs corpus/movie_conversations.txt'

    chatbot = Chatbot(movie_lines_path, movie_conversations_path)
    chatbot.fine_tune()

if __name__ == '__main__':
    main()