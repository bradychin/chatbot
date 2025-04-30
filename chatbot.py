#--------- Import libraries ---------#
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

#--------- Chatbot ---------#
def chatbot(prompt, model, tokenizer, device):
    formatted_prompt = f'{tokenizer.bos_token} {prompt} {tokenizer.eos_token}'
    encoded_prompt = tokenizer.encode(formatted_prompt, return_tensors='pt').to(device)

    output_sequences = model.generate(input_ids=encoded_prompt,
                                      max_length=100,
                                      temperature=0.8,
                                      top_k=40,
                                      top_p=0.9,
                                      do_sample=True,
                                      num_return_sequences=1,
                                      pad_token_id=tokenizer.pad_token_id,
                                      eos_token_id=tokenizer.eos_token_id)

    complete_response = tokenizer.decode(output_sequences[0], skip_special_tokens=False)
    input_text = tokenizer.decode(encoded_prompt[0], skip_special_tokens=False)
    response_text = complete_response[len(input_text):]

    if tokenizer.eos_token in response_text:
        response_text = response_text[:response_text.find(tokenizer.eos_token)]

    if prompt == 'stop':
        response_text = 'Have a good day!'

    print(f'Chatbot: {response_text}\n')

    return encoded_prompt, prompt

def debug(model, tokenizer, encoded_prompt):
    print(f'Encoded prompt: {encoded_prompt}')
    print(f'Decoded tokens: {tokenizer.convert_ids_to_tokens(encoded_prompt[0])}')


    with torch.no_grad():
        outputs = model(encoded_prompt)

    logits = outputs.logits
    last_token_logits = logits[0, -1, :]
    top_token_ids = torch.topk(last_token_logits, 10).indices

    print("\nTop 10 predicted tokens after EOS:")
    for token_id in top_token_ids:
        token = tokenizer.convert_ids_to_tokens(token_id.item())
        print(
            f"  {token_id.item()}: {token} (probability: {torch.softmax(last_token_logits, dim=0)[token_id.item()]:.4f})")

def main():
    model = GPT2LMHeadModel.from_pretrained('chatbot_model')
    tokenizer = GPT2Tokenizer.from_pretrained('chatbot_model')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    print('\nYou can talk with this chatbot. If you want to stop the conversation you can type "stop".\n')
    while True:
        prompt = input('>>> ')
        chatbot(prompt, model, tokenizer, device)

        if prompt == 'stop':
            break

    # Uncomment for debugging
    # debug(model, tokenizer, chatbot(model, tokenizer, device))

if __name__ == '__main__':
    main()