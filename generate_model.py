#--------- Import libraries ---------#
import sys
import pandas as pd
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

#--------- Process Data ---------#
class ProcessData:
    def __init__(self, personachat_file_path, train_val_split=0.2):
        self.personachat_file_path = personachat_file_path
        self.train_val_split = train_val_split

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        special_tokens = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
        self.tokenizer.add_special_tokens(special_tokens)

        try:
            self.dialog_data = self.load_personachat_data()
            logger.info(f'Created {len(self.dialog_data)} dialog paris.')
        except FileNotFoundError:
            logger.error(f'File "{personachat_file_path}" not found.')
            sys.exit(1)

        self.train_dataset, self.val_dataset = self.create_datasets()

    def preprocess_lines(self, lines):
        lines = lines.lower()
        lines = re.sub(r'\s+', ' ', lines).strip()
        lines = re.sub(r'[^\w\s.,?!\'"-]', '', lines)
        lines = re.sub(r'[""]', '"', lines)
        lines = re.sub(r"['']", "'", lines)
        return lines

    # Create dialog
    def load_personachat_data(self):
        logger.info(f'Loading data from {self.personachat_file_path}')
        df = pd.read_csv(self.personachat_file_path)
        dialog_data = []

        for _, row in df.iterrows():
            chat_text = row['chat']

            if not isinstance(chat_text, str) or not chat_text.strip():
                continue

            # Split chat into messages
            current_messages = chat_text.split('\n')
            current_messages = [message for message in current_messages if message.strip()]

            if len(current_messages) < 2:
                continue

            for i in range(0, len(current_messages) - 1, 2):
                if i + 1 >= len(current_messages):
                    continue

                prompt = self.preprocess_lines(current_messages[i])
                response = self.preprocess_lines(current_messages[i + 1])

                if 3 <= len(prompt.split()) <= 50 and 3 <= len(response.split()) <= 50:
                    dialog_data.append((prompt, response))

            if len(current_messages) >= 4:
                for start_idx in range(0, min(len(current_messages) - 3, 2)):
                    end_idx = min(start_idx + 8, len(current_messages))
                    if end_idx - start_idx < 4:
                        continue

                    context = []
                    for i in range(start_idx, end_idx):
                        text = self.preprocess_lines(current_messages[i])

                        if len(text.split()) > 50:
                            continue

                        context.append(text)

                    if len(context) >= 4:
                        dialog_data.append(context)
        logger.info(f'Created {len(dialog_data)} training examples.')
        return dialog_data

    # Create datasets
    def create_datasets(self):
        full_dataset = DialogManagement(self.dialog_data, self.tokenizer)
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
    def __init__(self, personachat_file_path):
        self.data_processor = ProcessData(personachat_file_path)
        self.tokenizer = self.data_processor.tokenizer
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def fine_tune(self):
        epochs = 4
        batch_size = 4
        patience = 1
        epochs_without_improvement = 0
        gradient_accumulation = 2

        train_loader, validation_loader = self.data_processor.get_dataloaders(batch_size=batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {device}\n')

        self.model.to(device)
        optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)

        training_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * training_steps),
            num_training_steps=training_steps
        )

        scaler = GradScaler() if device.type == 'cuda' else None

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
                        loss = outputs.loss / gradient_accumulation
                    scaler.scale(loss).backward()
                    if (batch_index + 1) % gradient_accumulation == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                else:
                    outputs = self.model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         labels=labels)
                    loss = outputs.loss / gradient_accumulation
                    loss.backward()
                    if (batch_index + 1) % gradient_accumulation == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                total_train_loss += loss.item() * gradient_accumulation

                if batch_index % 50 == 0:
                    logger.info(f'Epoch: {epoch}, Batch {batch_index}, Loss: {loss.item() * gradient_accumulation}')
                if batch_index % 250 == 0 and batch_index > 0:
                    checkpoint_chatbot_model_path = f'models/checkpoints/checkpoint_epoch_{epoch}_batch_{batch_index}_lr2e-5'
                    self.model.save_pretrained(checkpoint_chatbot_model_path)
                    self.tokenizer.save_pretrained(checkpoint_chatbot_model_path)
                    print(f'Saved checkpoint model at epoch {epoch} batch {batch_index}')

            if (batch_index + 1) % gradient_accumulation != 0:
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

            if average_validation_loss < best_validation_loss - 0.5:
                best_validation_loss = average_validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                best_model_path = f'models/val loss/model_best_val_loss_{best_validation_loss}'
                self.model.save_pretrained(best_model_path)
                self.tokenizer.save_pretrained(best_model_path)
                print(f'New best model saved with validation loss: {best_validation_loss}')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f'Early stopping at epoch {epoch}. No improvement for {patience} consecutive epochs.')
                    break

        chatbot_model_path = 'models/model'
        self.model.save_pretrained(chatbot_model_path)
        self.tokenizer.save_pretrained(chatbot_model_path)
        print('\nSaved:')
        print(f'Final model: {chatbot_model_path}.')
        print(f'Best model: {best_model_path}.')
        print(f'    > Epoch: {best_epoch}')
        print(f'    > Validation Loss: {best_validation_loss}.')


#--------- Main Function ---------#
def main():
    personachat_file_path = 'data/personachat.csv'

    model = GenerateModel(personachat_file_path)
    model.fine_tune()

if __name__ == '__main__':
    main()