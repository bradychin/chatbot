#--------- Import libraries ---------#
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
import torch
import logging
from torch.cuda.amp import autocast, GradScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#--------- Tokenize ---------#
class DialogManagement(Dataset):
    def __init__(self, dialog_pairs, tokenizer, max_length=128):
        self.dialog_pairs = dialog_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialog_pairs)

    def __getitem__(self, idx):
        prompt, response = self.dialog_pairs[idx]
        # Format: <BOS> prompt <EOS> response <EOS>
        combined_text = f"{self.tokenizer.bos_token} Human: {prompt} {self.tokenizer.eos_token}{response} {self.tokenizer.eos_token}"

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
                if len(parts) >= 5:
                    line_id, text = parts[0], parts[4]
                    text = text.strip()
                    if text:
                        lines[line_id] = text
        return lines

    # Import conversations
    def load_conversations(self):
        conversations = []
        with open(self.movie_conversations_path, 'r', encoding='iso-8859-1') as file:
            for i, line in enumerate(file):
                if i >= 400:
                    break
                parts = line.strip().split(' +++$+++ ')
                if len(parts) >= 4:
                    line_ids = eval(parts[3])
                    if len(line_ids) >= 2:
                        conversations.append(line_ids)
        return conversations

    # Create dialog
    def create_dialog(self):
        dialog = []
        for conversation in self.conversations:
            for i in range(len(conversation)-1):
                if conversation[i] in self.lines and conversation[i+1] in self.lines:
                    prompt = self.lines[conversation[i]]
                    response = self.lines[conversation[i+1]]
                    if 3 <= len(prompt.split()) <= 50 and 3 <= len(response.split()) <= 50:
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
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def fine_tune(self):
        epochs = 5
        batch_size = 3
        dataset = self.data_processor.get_dataset()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {device}')

        self.model.to(device)
        optimizer = AdamW(self.model.parameters(), lr=3e-5)
        training_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1*training_steps),
            num_training_steps=training_steps
        )

        scaler = GradScaler() if device.type == 'cuda' else None
        accumulation = 2

        # Fine tune
        self.model.train()
        optimizer.zero_grad()
        total_loss = 0
        for epoch in range(epochs):
            for batch_index, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                if input_ids.size(0) == 0:
                    print('Empty batch. Skipping')
                    continue

                if device.type == 'cuda':
                    with autocast():
                        outputs = self.model(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             labels=labels)
                        loss = outputs.loss / accumulation
                    scaler.scale(loss).backward()
                    if (batch_index + 1) % accumulation == 0:
                        scaler.unscale(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                else:
                    outputs = self.model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         labels=labels)
                    loss = outputs.loss / accumulation
                    loss.backward()
                    if (batch_index + 1) % accumulation == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                total_loss += loss.item() * accumulation

                if batch_index % 50 == 0:
                    logger.info(f'Epoch: {epoch}, Batch {batch_index}, Loss: {loss.item() * accumulation}')

            if (batch_index + 1) % accumulation != 0:
                if device.type == 'cuda':
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            avg_loss = total_loss / len(dataloader)
            print(f'Epoch: {epoch} Completed. Average Loss: {avg_loss}')

            # self.model.save_pretrained(f'fine_tuned_chatbot_epoch_{epoch}')
            # self.tokenizer.save_pretrained(f'fine_tuned_chatbot_epoch_{epoch}')
            # print('Saved checkpoint model.')

        self.model.save_pretrained('chatbot_model')
        self.tokenizer.save_pretrained("chatbot_model")
        print('Saved final model')

#--------- Main Function ---------#
def main():
    movie_lines_path = 'cornell movie-dialogs corpus/movie_lines.txt'
    movie_conversations_path = 'cornell movie-dialogs corpus/movie_conversations.txt'

    chatbot = Chatbot(movie_lines_path, movie_conversations_path)
    chatbot.fine_tune()

if __name__ == '__main__':
    main()