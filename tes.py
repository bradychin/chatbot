# --------- Import libraries ---------#
import pandas as pd
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
import torch
import logging
import re
from torch.cuda.amp import autocast, GradScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------- Tokenize ---------#
class DialogDataset(Dataset):
    """Dataset for training on multi-turn dialog contexts."""

    def __init__(self, dialog, tokenizer, max_length=128):
        self.dialog = dialog
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialog)

    def __getitem__(self, idx):
        """Process a multi-turn conversation context."""
        messages = self.dialog[idx]

        # Start with BOS token
        combined_text = f'{self.tokenizer.bos_token}'

        # Add alternating messages, with line breaks between turns
        for i in range(len(messages) - 1):
            combined_text += f'{messages[i]}'

            # Add newline between turns
            if i < len(messages) - 1:
                combined_text += '\n'

        # End with EOS token
        combined_text += f'{self.tokenizer.eos_token}'

        # Encode for the model
        encodings = self.tokenizer(
            combined_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()

        # For GPT training, labels are the same as inputs
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids
        }


# --------- Process Data ---------#
class DataProcessor:
    """Process and prepare dialog data for training."""

    def __init__(self, data_file_path, train_val_split=0.2):
        self.data_file_path = data_file_path
        self.train_val_split = train_val_split

        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        special_tokens = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
        self.tokenizer.add_special_tokens(special_tokens)

        # Load data
        try:
            self.dialog_data = self._load_dialog_data()
            logger.info(f'Created {len(self.dialog_data)} dialog pairs.')
        except FileNotFoundError:
            logger.error(f'File "{data_file_path}" not found.')
            raise

        # Create datasets
        self.train_dataset, self.val_dataset = self._create_datasets()

    def _preprocess_text(self, text):
        """Clean and standardize text."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s.,?!\'"-]', '', text)
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r"['']", "'", text)
        return text

    def _load_dialog_data(self):
        """Load and process dialog data from CSV file, converting to multi-turn format."""
        logger.info(f'Loading data from {self.data_file_path}')
        df = pd.read_csv(self.data_file_path)
        dialog_contexts = []

        for _, row in df.iterrows():
            chat_text = row['chat']

            # Skip invalid entries
            if not isinstance(chat_text, str) or not chat_text.strip():
                continue

            # Split chat into messages
            messages = [m.strip() for m in chat_text.split('\n') if m.strip()]

            if len(messages) < 2:
                continue

            # Process messages
            clean_messages = [self._preprocess_text(msg) for msg in messages]

            # Filter out messages that are too long
            clean_messages = [msg for msg in clean_messages if len(msg.split()) <= 50]

            # Create sliding windows of conversations (for context)
            # Each window will be a separate training example
            if len(clean_messages) >= 2:
                # Create windows of different sizes (2 to 8 messages)
                for window_size in range(2, min(9, len(clean_messages) + 1)):
                    for start_idx in range(0, len(clean_messages) - window_size + 1):
                        window = clean_messages[start_idx:start_idx + window_size]

                        # Make sure we have valid messages
                        if all(len(msg.split()) >= 3 for msg in window):
                            dialog_contexts.append(window)

        logger.info(f'Created {len(dialog_contexts)} multi-turn dialog contexts.')
        return dialog_contexts

    def _create_datasets(self):
        """Split data into training and validation sets."""
        full_dataset = DialogDataset(self.dialog_data, self.tokenizer)
        validation_size = int(len(full_dataset) * self.train_val_split)
        train_size = len(full_dataset) - validation_size
        train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])

        logger.info(f'Training dataset: {train_size} dialog pairs.')
        logger.info(f'Validation dataset: {validation_size} dialog pairs.\n')

        return train_dataset, validation_dataset

    def get_dataloaders(self, batch_size=4):
        """Create data loaders for training and validation."""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        validation_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        return train_loader, validation_loader


# --------- Model Trainer ---------#
class ModelTrainer:
    """Train a GPT-2 model on dialog data."""

    def __init__(self, data_file_path):
        self.data_processor = DataProcessor(data_file_path)
        self.tokenizer = self.data_processor.tokenizer
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def _save_checkpoint(self, path):
        """Save model and tokenizer checkpoint."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f'Model saved to {path}')

    def fine_tune(self, epochs=4, batch_size=8, patience=1, gradient_accumulation=2):
        """Fine-tune the model on dialog data."""
        train_loader, validation_loader = self.data_processor.get_dataloaders(batch_size=batch_size)

        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {device}\n')
        self.model.to(device)

        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=5e-6, weight_decay=0.01)
        training_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * training_steps),
            num_training_steps=training_steps
        )

        # Setup mixed precision training if on CUDA
        scaler = GradScaler() if device.type == 'cuda' else None

        # Training state variables
        best_validation_loss = float('inf')
        best_epoch = -1
        epochs_without_improvement = 0

        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            optimizer.zero_grad()

            for batch_index, batch in enumerate(train_loader):
                try:
                    # Prepare batch
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    if input_ids.size(0) == 0:
                        logger.warning('Empty batch. Skipping')
                        continue

                    # Forward pass with mixed precision
                    if device.type == 'cuda':
                        with autocast():
                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels
                            )
                            loss = outputs.loss / gradient_accumulation

                        # Backward pass with gradient scaling
                        scaler.scale(loss).backward()

                        # Update on gradient accumulation step
                        if (batch_index + 1) % gradient_accumulation == 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            scheduler.step()
                            optimizer.zero_grad()
                    else:
                        # Standard training for CPU
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss / gradient_accumulation
                        loss.backward()

                        # Update on gradient accumulation step
                        if (batch_index + 1) % gradient_accumulation == 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()

                    # Track loss
                    total_train_loss += loss.item() * gradient_accumulation

                    # Logging
                    if batch_index % 50 == 0:
                        logger.info(f'Epoch: {epoch}, Batch {batch_index}, Loss: {loss.item() * gradient_accumulation}')

                    # Periodic checkpoint saving
                    if batch_index % 250 == 0 and batch_index > 0:
                        checkpoint_path = f'models/checkpoints/checkpoint_epoch_{epoch}_batch_{batch_index}'
                        self._save_checkpoint(checkpoint_path)

                except Exception as e:
                    logger.error(f"Error in training batch {batch_index}: {e}")
                    continue

            # Handle final step if needed
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

            # Calculate average training loss
            average_train_loss = total_train_loss / len(train_loader)
            logger.info(f'Epoch: {epoch} Completed. Average Loss: {average_train_loss}')

            # Validation phase
            self.model.eval()
            total_validation_loss = 0

            with torch.no_grad():
                for validation_batch in validation_loader:
                    try:
                        # Prepare batch
                        validation_input_ids = validation_batch['input_ids'].to(device)
                        validation_attention_mask = validation_batch['attention_mask'].to(device)
                        validation_labels = validation_batch['labels'].to(device)

                        if validation_input_ids.size(0) == 0:
                            continue

                        # Forward pass
                        validation_outputs = self.model(
                            input_ids=validation_input_ids,
                            attention_mask=validation_attention_mask,
                            labels=validation_labels
                        )

                        # Track loss
                        validation_loss = validation_outputs.loss
                        total_validation_loss += validation_loss.item()

                    except Exception as e:
                        logger.error(f"Error in validation batch: {e}")
                        continue

            # Calculate average validation loss
            average_validation_loss = total_validation_loss / len(validation_loader)
            logger.info(f'Epoch: {epoch} Validation Loss: {average_validation_loss}')

            # Check for improvement
            if average_validation_loss < best_validation_loss - 0.5:
                best_validation_loss = average_validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0

                # Save best model
                best_model_path = f'models/best_model_val_loss_{best_validation_loss:.4f}'
                self._save_checkpoint(best_model_path)
                logger.info(f'New best model saved with validation loss: {best_validation_loss:.4f}')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f'Early stopping at epoch {epoch}. No improvement for {patience} consecutive epochs.')
                    break

        # Save final model
        final_model_path = 'models/final_model'
        self._save_checkpoint(final_model_path)

        # Report results
        logger.info('\nTraining completed:')
        logger.info(f'Final model saved to: {final_model_path}')
        logger.info(f'Best model (epoch {best_epoch}) saved to: models/best_model_val_loss_{best_validation_loss:.4f}')
        logger.info(f'Best validation loss: {best_validation_loss:.4f}')


# --------- Main Function ---------#
def main():
    """Train a chatbot model on dialog data."""
    data_file_path = 'data/personachat_full.csv'

    try:
        trainer = ModelTrainer(data_file_path)
        trainer.fine_tune()
    except Exception as e:
        logger.error(f"Training failed: {e}")


if __name__ == '__main__':
    main()