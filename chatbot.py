#--------- Import libraries ---------#
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

#--------- Chatbot ---------#
model = GPT2LMHeadModel.from_pretrained('fine_tuned_chatbot')
tokenizer = GPT2Tokenizer.from_pretrained('fine_tuned_chatbot')
model.config.pad_token_id = tokenizer.pad_token_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

prompt = 'Hello, there'

encoded_prompt = tokenizer.encode(f'{tokenizer.bos_token} {prompt} {tokenizer.eos_token}',
                                  return_tensors='pt').to(device)

# output_sequences = model.generate(
#             input_ids=encoded_prompt,
#             max_length=100 + len(encoded_prompt[0]),
#             temperature=0.8,
#             top_k=50,
#             top_p=0.9,
#             do_sample=True,
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id
#         )

output_sequences = model.generate(
    input_ids=encoded_prompt,
    max_length=100 + len(encoded_prompt[0]),
    num_beams=1,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)

generated = output_sequences[0][len(encoded_prompt[0]):]
response = tokenizer.decode(generated)

print(tokenizer.pad_token_id)
print(model.config.pad_token_id)
with torch.no_grad():
    outputs = model(encoded_prompt)
logits = outputs.logits
print(logits.shape)
print(torch.argmax(logits[0, -1, :]))

print(f'Prompt: {prompt}')
print(f'Response: {response}')