# --------- 1. Import libraries ---------#
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
import torch
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------- 2. Process Data ---------#
class MovieDialogDataset(Dataset):
    def __init__(self, dialog_pairs, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.dialog_pairs = dialog_pairs
        self.max_length = max_length

        # Add special tokens
        special_tokens = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
        self.tokenizer.add_special_tokens(special_tokens)

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


class ProcessData:
    def __init__(self, movie_lines_path, movie_conversations_path):
        self.movie_lines_path = movie_lines_path
        self.movie_conversations_path = movie_conversations_path
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.lines = self.load_lines()
        self.conversations = self.load_conversations()
        self.dialog = self.create_dialog()

    # Import lines
    def load_lines(self):
        lines = {}
        try:
            with open(self.movie_lines_path, 'r', encoding='iso-8859-1') as file:
                for line in file:
                    parts = line.strip().split(' +++$+++ ')
                    if len(parts) >= 5:  # Ensure the line has the expected format
                        line_id, text = parts[0], parts[4]
                        lines[line_id] = text
                    else:
                        logger.warning(f"Skipping malformed line: {line}")
        except Exception as e:
            logger.error(f"Error loading lines: {e}")

        logger.info(f"Loaded {len(lines)} lines")
        return lines

    # Import conversations
    def load_conversations(self):
        conversations = []
        try:
            with open(self.movie_conversations_path, 'r', encoding='iso-8859-1') as file:
                for line in file:
                    parts = line.strip().split(' +++$+++ ')
                    if len(parts) >= 4:  # Ensure the line has the expected format
                        try:
                            line_ids = eval(parts[3])
                            conversations.append(line_ids)
                        except Exception as e:
                            logger.warning(f"Skipping conversation with invalid format: {e}")
                    else:
                        logger.warning(f"Skipping malformed conversation line: {line}")
        except Exception as e:
            logger.error(f"Error loading conversations: {e}")

        logger.info(f"Loaded {len(conversations)} conversations")
        return conversations

    # Create dialog
    def create_dialog(self):
        dialog = []
        for conversation in self.conversations:
            for i in range(len(conversation) - 1):
                if conversation[i] in self.lines and conversation[i + 1] in self.lines:
                    prompt = self.lines[conversation[i]]
                    response = self.lines[conversation[i + 1]]
                    dialog.append((prompt, response))

        logger.info(f"Created {len(dialog)} dialogue pairs")
        return dialog

    def get_dataset(self):
        return MovieDialogDataset(self.dialog, self.tokenizer)


# --------- 3. Chatbot ---------#
class Chatbot:
    def __init__(self, movie_lines_path, movie_conversations_path):
        self.data_processor = ProcessData(movie_lines_path, movie_conversations_path)
        self.tokenizer = self.data_processor.tokenizer
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

        # Resize token embeddings to account for added special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

    def fine_tune(self, batch_size=4, epochs=3, learning_rate=5e-5):
        # Get dataset
        dataset = self.data_processor.get_dataset()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Setup training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        self.model.to(device)
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Add learning rate scheduler
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=total_steps
        )

        # Fine tune
        self.model.train()
        total_loss = 0

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                self.model.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                epoch_loss += loss.item()

                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                if batch_idx % 10 == 0:
                    logger.info(
                        f"Epoch: {epoch + 1}/{epochs}, Batch: {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

            avg_epoch_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")

        # Save model and tokenizer
        output_dir = "fine_tuned_chatbot"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def generate_response(self, prompt, max_length=100):
        """Generate a response to the given prompt"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        # Tokenize the prompt
        encoded_prompt = self.tokenizer.encode(
            f"{self.tokenizer.bos_token} {prompt} {self.tokenizer.eos_token}",
            return_tensors="pt"
        ).to(device)

        # Generate response
        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            max_length=max_length + len(encoded_prompt[0]),
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Decode response
        generated = output_sequences[0][len(encoded_prompt[0]):]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)

        return response


# --------- Main Function ---------#
def main():
    movie_lines_path = 'test_lines.txt'
    # movie_lines_path = 'cornell movie-dialogs corpus/movie_lines.txt'
    movie_conversations_path = 'test_conversations.txt'
    # movie_conversations_path = 'cornell movie-dialogs corpus/movie_conversations.txt'

    chatbot = Chatbot(movie_lines_path, movie_conversations_path)
    chatbot.fine_tune(batch_size=4, epochs=3)

    # Test the chatbot
    test_prompts = [
        "Hello, how are you?",
        "What's your favorite movie?"
    ]

    for prompt in test_prompts:
        response = chatbot.generate_response(prompt)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("-" * 50)


if __name__ == '__main__':
    main()