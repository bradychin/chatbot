#--------- 1. Import libraries ---------#
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

#--------- 2. Process Data ---------#
class ProcessData:
    def __init__(self, movie_lines_path, movie_conversations_path):
        self.movie_lines_path = movie_lines_path
        self.movie_conversations_path = movie_conversations_path
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.lines = self.load_lines()
        self.conversations = self.load_conversations()
        self.dialog = self.create_dialog()
        self.dialog_tokens = self.convert_to_tokens()

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

    def convert_to_tokens(self):
        dialog_tokens = []
        for pair in self.dialog:
            input_token = self.tokenizer.encode(pair[0], return_tensors='pt', max_length=1024, truncation=True)
            output_token = self.tokenizer.encode(pair[1], return_tensors='pt', max_length=1024, truncation=True)
            print(f'Input: {input_token.shape}')
            print(f'Output: {output_token.shape}')
            concat_token = torch.cat((input_token, output_token), dim=1)
            print('Concat success')
            dialog_tokens.append(concat_token)
        return dialog_tokens

#--------- 3. Chatbot ---------#
class Chatbot:
    def __init__(self, movie_lines_path, movie_conversations_path):
        self.data_processor = ProcessData(movie_lines_path, movie_conversations_path)
        self.pretrained_model = GPT2LMHeadModel.from_pretrained('gpt2')

    def fine_tune(self):
        print('fine tune function')
        input_ids = torch.cat(self.data_processor.dialog_tokens, dim=0)
        print('input_ids success')
        dataset = TensorDataset(input_ids)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        print('dataloader success')
        optimizer = AdamW(self.pretrained_model.parameters(), lr=5e-6)

        # Fine tune
        self.pretrained_model.train()
        print('train_success')
        epochs = 3
        for epoch in range(epochs):
            for batch in dataloader:
                inputs = batch[0]
                outputs = self.pretrained_model(input_ids=inputs, labels=inputs)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f'Epoch: {epoch}. Loss: {loss.item()}')

        self.pretrained_model.save_pretrained("fine_tuned_chatbot")
        self.data_processor.tokenizer.save_pretrained("fine_tuned_chatbot")

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