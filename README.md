# Generative Chatbot Project

This project implements a simple, multi-turn generative chatbot using Natural Language Processing (NLP). The chatbot is designed for casual conversation with a friendly and sometimes sarcastic tone. It is built using a pre-trained GPT model, fine-tuned on the Cornell Movie Dialogue Corpus to facilitate coherent, engaging dialogue.

## Features
- Responses generated based on context from previous dialogue.
- Built for text-based interactions only.

## Technologies Used
- **Python**: Main programming language.
- **Hugging Face Transformers**: Pre-trained models (GPT-2) for response generation.
- **PyTorch/TensorFlow**: Frameworks used for model fine-tuning and training.

## Project Overview
1. **Dataset**: The model is fine-tuned on the [PersonaChat Dataset](https://www.kaggle.com/datasets/atharvjairath/personachat/data), which contains dialogue agents.
2. **Model**: GPT-2/3 (or equivalent) pre-trained model, fine-tuned with the dataset.
3. **Conversation Management**: Simple memory system to track the flow of conversation, providing context for more accurate responses.
4. **Interface**: Basic text-based interface for interaction with the chatbot.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/bradychin/chatbot.git
    cd chatbot
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the pre-trained model (Hugging Face or another source).

## Usage

To interact with the chatbot, run the script:

```bash
python main.py
```
If no trained model exists at the configured path, training will begin.

If a trained model exists, the chatbot will launch immediately.
