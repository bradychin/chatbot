# --------- Import libraries ---------#
import sys
import os
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
import torch
import logging
from torch.cuda.amp import autocast, GradScaler
import re
import numpy as np
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------- Tokenize ---------#
class DialogManagement(Dataset):
    def __init__(self, dialog_pairs, tokenizer, max_length=256):
        self.dialog_pairs = dialog_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialog_pairs)

    def __getitem__(self, idx):
        conversation = self.dialog_pairs[idx]

        # Format the entire conversation with multiple turns if available
        if isinstance(conversation, list) and len(conversation) > 1:
            # For multi-turn conversations
            combined_text = f"{self.tokenizer.bos_token} "

            for i, (speaker, text) in enumerate(conversation):
                prefix = "User: " if speaker == "user" else "Bot: "
                combined_text += f"{prefix}{text}"

                # Add newline except after the last turn
                if i < len(conversation) - 1:
                    combined_text += "\n"

            combined_text += f" {self.tokenizer.eos_token}"
        else:
            # For single prompt-response pairs (backward compatibility)
            prompt, response = conversation if isinstance(conversation, tuple) else (
            conversation[0][1], conversation[1][1])
            combined_text = f"{self.tokenizer.bos_token} User: {prompt}\nBot: {response} {self.tokenizer.eos_token}"

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


# --------- Process Data ---------#
class ProcessData:
    def __init__(self, movie_lines_path, movie_conversations_path, train_val_split=0.2):
        self.movie_lines_path = movie_lines_path
        self.movie_conversations_path = movie_conversations_path
        self.train_val_split = train_val_split

        # Create directories for model checkpoints if they don't exist
        os.makedirs('models/checkpoints', exist_ok=True)
        os.makedirs('models/val loss', exist_ok=True)

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        special_tokens = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
        self.tokenizer.add_special_tokens(special_tokens)

        try:
            self.lines = self.load_lines()
            logger.info(f'Loaded {len(self.lines)} lines.')
        except FileNotFoundError:
            logger.error(
                'File "movie_lines.txt" not found. \nExpected in file path: cornell movie-dialogs corpus/movie_lines.txt')
            sys.exit(1)

        try:
            self.conversations = self.load_conversations()
            logger.info(f'Loaded {len(self.conversations)} conversations.')
        except FileNotFoundError:
            logger.error(
                'File "movie_conversations.txt" not found. \nExpected in file path: cornell movie-dialogs corpus/movie_conversations.txt')
            sys.exit(1)

        self.dialog_data = self.create_dialog()
        logger.info(f'Created {len(self.dialog_data)} dialog items.')

        self.train_dataset, self.val_dataset = self.create_datasets()

    def preprocess_lines(self, lines):
        """Clean and normalize text"""
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
                    line_id, character_id, text = parts[0], parts[1], parts[4]
                    text = self.preprocess_lines(text.strip())
                    if text:
                        lines[line_id] = {"text": text, "character_id": character_id}
        return lines

    # Import conversations
    def load_conversations(self):
        conversations = []
        with open(self.movie_conversations_path, 'r', encoding='iso-8859-1') as file:
            for i, line in enumerate(file):
                # Uncomment to use a subset of dataset for testing
                number_of_lines = 6000
                if i >= number_of_lines:
                    break

                parts = line.strip().split(' +++$+++ ')
                if len(parts) >= 4:
                    line_ids = eval(parts[3])
                    if len(line_ids) >= 2:
                        conversations.append(line_ids)
        return conversations

    # Create dialog data with context
    def create_dialog(self):
        dialog_data = []

        for conversation in self.conversations:
            if len(conversation) < 2:
                continue

            # For each conversation, create both pair-wise samples and multi-turn samples
            valid_lines = []

            for line_id in conversation:
                if line_id in self.lines:
                    valid_lines.append(line_id)

            if len(valid_lines) < 2:
                continue

            # Create pair-wise samples (traditional approach)
            for i in range(len(valid_lines) - 1):
                prompt_id = valid_lines[i]
                response_id = valid_lines[i + 1]

                prompt = self.lines[prompt_id]["text"]
                response = self.lines[response_id]["text"]

                # Filter out very short or very long utterances
                if 3 <= len(prompt.split()) <= 50 and 3 <= len(response.split()) <= 50:
                    # For simplicity, alternate user/bot roles in the conversation
                    # In a real implementation, you might want to track speaker identity
                    dialog_data.append((("user", prompt), ("bot", response)))

            # Create multi-turn samples (for better context understanding)
            # Take chunks of the conversation with increasing context
            # For example: turns 1-3, turns 1-4, turns 1-5, etc.
            if len(valid_lines) >= 3:
                for end_idx in range(2, min(len(valid_lines), 6)):  # Max 5 turns for context
                    context = []
                    for i in range(end_idx + 1):
                        line_id = valid_lines[i]
                        text = self.lines[line_id]["text"]

                        # Skip if utterance is too long
                        if len(text.split()) > 50:
                            continue

                        # Assign roles alternating between user and bot
                        role = "user" if i % 2 == 0 else "bot"
                        context.append((role, text))

                    # Only add if we have at least 3 turns (to provide actual context)
                    if len(context) >= 3:
                        dialog_data.append(context)

        return dialog_data

    # Create datasets
    def create_datasets(self):
        full_dataset = DialogManagement(self.dialog_data, self.tokenizer)
        validation_size = int(len(full_dataset) * self.train_val_split)
        train_size = len(full_dataset) - validation_size
        train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])
        logger.info(f'Training dataset: {train_size} dialog items.')
        logger.info(f'Validation dataset: {validation_size} dialog items.\n')
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


# --------- Model ---------#
class GenerateModel:
    def __init__(self, movie_lines_path, movie_conversations_path):
        self.data_processor = ProcessData(movie_lines_path, movie_conversations_path)
        self.tokenizer = self.data_processor.tokenizer
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def fine_tune(self):
        epochs = 4
        batch_size = 8
        patience = 2  # Increased patience
        epochs_without_improvement = 0
        gradient_accumulation_steps = 2

        train_loader, validation_loader = self.data_processor.get_dataloaders(batch_size=batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {device}\n')

        self.model.to(device)
        optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)  # Added weight decay

        training_steps = len(train_loader) * epochs // gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * training_steps),
            num_training_steps=training_steps
        )

        scaler = GradScaler() if device.type == 'cuda' else None

        best_validation_loss = float('inf')
        best_epoch = -1

        # Save training config
        training_config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup": "10% of training steps",
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open('models/training_config.json', 'w') as f:
            json.dump(training_config, f, indent=4)

        # Fine tune
        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.model.train()
            optimizer.zero_grad()
            total_train_loss = 0
            steps = 0

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
                        loss = outputs.loss / gradient_accumulation_steps
                    scaler.scale(loss).backward()

                    if (batch_index + 1) % gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                        steps += 1
                else:
                    outputs = self.model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         labels=labels)
                    loss = outputs.loss / gradient_accumulation_steps
                    loss.backward()

                    if (batch_index + 1) % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        steps += 1

                total_train_loss += loss.item() * gradient_accumulation_steps

                # Log progress
                if batch_index % 50 == 0:
                    logger.info(
                        f'Epoch: {epoch}, Batch {batch_index}, Loss: {loss.item() * gradient_accumulation_steps:.4f}')

                # Save checkpoint periodically
                if batch_index % 250 == 0 and batch_index > 0:
                    checkpoint_path = f'models/checkpoints/checkpoint_epoch_{epoch}_batch_{batch_index}'
                    self.model.save_pretrained(checkpoint_path)
                    self.tokenizer.save_pretrained(checkpoint_path)
                    logger.info(f'Saved checkpoint model at epoch {epoch} batch {batch_index}')

            # Handle remaining gradients if needed
            if (batch_index + 1) % gradient_accumulation_steps != 0:
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
            epoch_time = time.time() - epoch_start_time
            logger.info(f'Epoch: {epoch} completed in {epoch_time:.2f}s. Average Loss: {average_train_loss:.4f}')

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
            logger.info(f'Epoch: {epoch} Validation Loss: {average_validation_loss:.4f}')

            # Save model if validation loss improved
            if average_validation_loss < best_validation_loss - 0.05:  # More lenient improvement threshold
                best_validation_loss = average_validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                best_model_path = f'models/val loss/model_best_val_loss_{best_validation_loss:.4f}'
                self.model.save_pretrained(best_model_path)
                self.tokenizer.save_pretrained(best_model_path)
                logger.info(f'New best model saved with validation loss: {best_validation_loss:.4f}')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f'Early stopping at epoch {epoch}. No improvement for {patience} consecutive epochs.')
                    break

        # Save final model
        chatbot_model_path = 'models/model'
        self.model.save_pretrained(chatbot_model_path)
        self.tokenizer.save_pretrained(chatbot_model_path)

        logger.info('\nTraining completed:')
        logger.info(f'Final model: {chatbot_model_path}')
        logger.info(f'Best model: best_model_path (Epoch: {best_epoch}, Validation Loss: {best_validation_loss:.4f})')

        # Update training config with results
        training_config.update({
            "completed": True,
            "best_epoch": best_epoch,
            "best_validation_loss": float(best_validation_loss),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        with open('models/training_config.json', 'w') as f:
            json.dump(training_config, f, indent=4)


# --------- Main Function ---------#
def main():
    movie_lines_path = 'cornell movie-dialogs corpus/movie_lines.txt'
    movie_conversations_path = 'cornell movie-dialogs corpus/movie_conversations.txt'

    model = GenerateModel(movie_lines_path, movie_conversations_path)
    model.fine_tune()


if __name__ == '__main__':
    main()