# main.py
import os
from train import train
from chatbot import chat


def main():
    if not os.path.exists('saved_models/chatbot_model.pth'):
        print("Model not found, starting training...")
        train()

    print("Starting chatbot...")
    chat()


if __name__ == '__main__':
    main()