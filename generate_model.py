#--------- Import libraries ---------#
import sys
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
import torch
import logging
from torch.cuda.amp import autocast, GradScaler
import re

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
    def __init__(self, movie_lines_path, movie_conversations_path, train_val_split=0.2):
        self.movie_lines_path = movie_lines_path
        self.movie_conversations_path = movie_conversations_path
        self.train_val_split = train_val_split

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        special_tokens = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
        self.tokenizer.add_special_tokens(special_tokens)
        try:
            self.lines = self.load_lines()
            logger.info(f'Loaded {len(self.lines)} lines.')
        except FileNotFoundError:
            logger.error('File "movie_lines.txt" not found. \nExpected in file path: cornell movie-dialogs corpus/movie_lines.txt')
            sys.exit(1)
        try:
            self.conversations = self.load_conversations()
            logger.info(f'Loaded {len(self.conversations)} conversations.')
        except FileNotFoundError:
            logger.error('File "movie_conversations.txt" not found. \nExpected in file path: cornell movie-dialogs corpus/movie_conversations.txt')
            sys.exit(1)
        self.dialog = self.create_dialog()
        logger.info(f'Created {len(self.dialog)} dialog pairs.')
        self.train_dataset, self.val_dataset = self.create_datasets()

    def preprocess_lines(self, lines):
        lines = lines.lower()
        lines = re.sub(r'\s+', ' ', lines).strip()
        lines = re.sub(r'[^\w\s.,?!\'"-]', '', lines)
        lines = re.sub(r'[""]', '"', lines)
        lines = re.sub(r"['']", "'", lines)
        return lines

    # Import lines
    def load_lines(self):
        lines = {}
        with open(self.movie_lines_path, 'r', encoding='iso-8859-1') as file:
            for line in file:
                parts = line.strip().split(' +++$+++ ')
                if len(parts) >= 5:
                    line_id, text = parts[0], parts[4]
                    text = self.preprocess_lines(text.strip())
                    if text:
                        lines[line_id] = text
        return lines

    # Import conversations
    def load_conversations(self):
        conversations = []
        with open(self.movie_conversations_path, 'r', encoding='iso-8859-1') as file:
            for i, line in enumerate(file):

                # Uncomment to use a subset of dataset for testing
                number_of_lines = 400
                if i >= number_of_lines:
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

    # Create datasets
    def create_datasets(self):
        full_dataset = DialogManagement(self.dialog, self.tokenizer)
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

#--------- Model ---------#
class GenerateModel:
    def __init__(self, movie_lines_path, movie_conversations_path):
        self.data_processor = ProcessData(movie_lines_path, movie_conversations_path)
        self.tokenizer = self.data_processor.tokenizer
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def fine_tune(self):
        epochs = 1
        batch_size = 3

        train_loader, validation_loader = self.data_processor.get_dataloaders(batch_size=batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {device}\n')

        self.model.to(device)
        optimizer = AdamW(self.model.parameters(), lr=3e-5)

        training_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1*training_steps),
            num_training_steps=training_steps
        )

        scaler = GradScaler() if device.type == 'cuda' else None
        accumulation = 2

        best_validation_loss = float('inf')
        best_epoch = -1

        # Fine tune
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            total_train_loss = 0

            for batch_index, batch in enumerate(train_loader):
                # Training stage
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                if input_ids.size(0) == 0:
                    logger.warning('Empty batch. Skipping')
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

                total_train_loss += loss.item() * accumulation

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

            average_train_loss = total_train_loss / len(train_loader)
            logger.info(f'Epoch: {epoch} Completed. Average Loss: {average_train_loss}')

            # Validation stage
            self.model.eval()
            total_validation_loss = 0

            with torch.no_grad():
                for validation_batch in validation_loader:
                    validation_input_ids = validation_batch['input_ids'].to(device)
                    validation_attention_mask = validation_batch['attention_mask'].to(device)
                    validation_labels = validation_batch['labels'].to(device)

                    if validation_input_ids.size(0) == 0:
                        continue

                    validation_outputs = self.model(input_ids=validation_input_ids,
                                                    attention_mask=validation_attention_mask,
                                                    labels=validation_labels)

                    validation_loss = validation_outputs.loss
                    total_validation_loss += validation_loss.item()

            average_validation_loss = total_validation_loss / len(validation_loader)
            logger.info(f'Epoch: {epoch} Validation Loss: {average_validation_loss}')

            if average_validation_loss < best_validation_loss:
                best_validation_loss = average_validation_loss
                best_epoch = epoch
                best_model_path = 'Models/chatbot_model_best'
                self.model.save_pretrained(best_model_path)
                self.tokenizer.save_pretrained(best_model_path)
                print(f'New best model saved with validation loss: {best_validation_loss}')

        chatbot_model_path = 'Models/chatbot_model'
        self.model.save_pretrained(chatbot_model_path)
        self.tokenizer.save_pretrained(chatbot_model_path)
        print('\nSaved:')
        print(f'Final model: {chatbot_model_path}.')
        print(f'Best model: {best_model_path}.')
        print(f'    > Epoch: {best_epoch}')
        print(f'    > Validation Loss: {best_validation_loss}.')


#--------- Main Function ---------#
def main():
    movie_lines_path = 'cornell movie-dialogs corpus/movie_lines.txt'
    movie_conversations_path = 'cornell movie-dialogs corpus/movie_conversations.txt'

    model = GenerateModel(movie_lines_path, movie_conversations_path)
    model.fine_tune()

if __name__ == '__main__':
    main()