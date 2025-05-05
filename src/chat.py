# --------- Import libraries ---------#
from src.log.logger import get_logger
logger = get_logger(__name__)

# --------- Response Generator ---------#
from src.utils.response_generator import ResponseGenerator

# --------- Chatbot ---------#
class Chatbot:
    def __init__(self, response_generator):
        self.response_generator = response_generator
        self.session_context = 'You are a helpful chatbot.'

    def generate_response(self, user_input):
        """Generate response for to the user input."""
        if not user_input.strip():
            return 'I did not receive an input'

        prompt = f'{self.session_context}\nUser: {user_input}\nBot:'
        response = self.response_generator.generate(prompt)

        if len(response.split()) < 3 or not any(char.isalpha() for char in response):
            response = 'I need to think about that. Could you ask me something else?'

        return response

# --------- Main ---------#
def start_chatbot(model_path):
    try:
        response_generator = ResponseGenerator(model_path)
        chatbot = Chatbot(response_generator)

        print('\nWelcome to the chatbot! Type "stop" at any time to end the conversation.\n')

        while True:
            try:
                user_input = input('>>> ')

                if user_input.lower() == "stop":
                    print("Chatbot: Have a good day!")
                    logger.info('Stopping chatbot...')
                    break

                response = chatbot.generate_response(user_input)
                print(f'Chatbot: {response}\n')

            except Exception as e:
                print(f"Error occurred: {e}")
                print("Restarting conversation...")
                continue

    except Exception as e:
        logger.error(f'Failed to initialize chatbot: {e}')
        print('Could not start chatbot.')

if __name__ == '__main__':
    from src import config
    start_chatbot(model_path=config.model_path)