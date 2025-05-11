#--------- Import libraries ---------#
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup
import torch
from torch.cuda.amp import autocast, GradScaler
from src.log.logger import get_logger
logger = get_logger(__name__)

#--------- Import Classes ---------#
from src.utils.data_processor import DataProcessor
from src import config

#--------- Model ---------#
class ModelTrainer:
    """Train GPT2 model."""
    def __init__(self, dataset_file_path):
        self.data_processor = DataProcessor(dataset_file_path)
        self.tokenizer = self.data_processor.tokenizer
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def _save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f'Model saved to {path}')

    def fine_tune(self):
        """Fine tune model on dialog data"""
        train_loader, validation_loader = self.data_processor.get_dataloaders(batch_size=config.batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {device}\n')

        self.model.to(device)
        optimizer = AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        training_steps = len(train_loader) * config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * training_steps),
            num_training_steps=training_steps
        )

        scaler = GradScaler() if device.type == 'cuda' else None

        best_validation_loss = float('inf')
        best_epoch = -1
        epochs_without_improvement = 0

        # Training loop
        for epoch in range(config.epochs):
            self.model.train()
            optimizer.zero_grad()
            total_train_loss = 0

            for batch_index, batch in enumerate(train_loader):
                try:
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
                            loss = outputs.loss / config.gradient_accumulation
                        scaler.scale(loss).backward()
                        if (batch_index + 1) % config.gradient_accumulation == 0:
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
                        loss = outputs.loss / config.gradient_accumulation
                        loss.backward()
                        if (batch_index + 1) % config.gradient_accumulation == 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()

                    total_train_loss += loss.item() * config.gradient_accumulation

                    if batch_index % 100 == 0:
                        print(f'Epoch: {epoch}, Batch {batch_index}, Loss: {loss.item() * config.gradient_accumulation}')
                    if batch_index % 1000 == 0 and batch_index > 0:
                        self._save_model(config.checkpoint_file_path.format(epoch=epoch, batch_index=batch_index))

                except Exception as e:
                    logger.error(f'Error in training batch {batch_index}:{e}')
                    continue

            if (batch_index + 1) % config.gradient_accumulation != 0:
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
                    try:
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
                    except Exception as e:
                        logger.error(f'Error in validation batch: {e}')
                        continue

            average_validation_loss = total_validation_loss / len(validation_loader)
            logger.info(f'Epoch: {epoch} Validation Loss: {average_validation_loss}')

            if average_validation_loss < best_validation_loss - 0.5:
                best_validation_loss = average_validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                self._save_model(config.best_model_file_path.format(best_validation_loss=best_validation_loss))
                print(f'New best model saved with validation loss: {best_validation_loss}')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    logger.info(f'Early stopping at epoch {epoch}. No improvement for {config.patience} consecutive epochs.')
                    break

        self._save_model(config.final_model_file_path)
        print('\nSaved:')
        print(f'Final model: {config.final_model_file_path}.')
        print(f'Best model: {config.best_model_file_path.format(best_validation_loss=best_validation_loss)}.')
        print(f'    > Epoch: {best_epoch}')
        print(f'    > Validation Loss: {best_validation_loss}.')

#--------- Main Function ---------#
def train_model():
    try:
        model = ModelTrainer(config.dataset_file_path)
        model.fine_tune()
    except Exception as e:
        logger.error(f'Training failed: {e}')

if __name__ == '__main__':
    train_model()