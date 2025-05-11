#--------- Import libraries ---------#
import os
from src.log.logger import get_logger
logger = get_logger(__name__)

#--------- Import Classes ---------#
from src.model_trainer import train_model
from src.chat import start_chatbot
from src.config import model_path

#--------- Main ---------#
def main():
    if not os.path.exists(model_path):
        logger.error(f'Model not found in {model_path}. Starting training...')
        train_model()

    logger.info('Starting chatbot...')
    start_chatbot(model_path)

if __name__ == '__main__':
    main()