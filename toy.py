import pickle
import os
from transformers import AutoTokenizer, AutoModel

test_text_path = "/public/home/CS172/tengzhh2022-cs172/finalproject/data/test/text"
test_motion_path = "/public/home/CS172/tengzhh2022-cs172/finalproject/data/test/motion"
tokenizer_path = "/public/home/CS172/tengzhh2022-cs172/finalproject/tokenizer/gpt2"

tokenizer_path = "/public/home/CS172/tengzhh2022-cs172/finalproject/tokenizer/contriever-msmarco"
model_path = "/public/home/CS172/tengzhh2022-cs172/finalproject/model/contriever-msmarco"


text_list = []
motion_list = []


tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModel.from_pretrained(model_path)
model.eval()

print(f'bos={tokenizer.bos_token_id} eos={tokenizer.eos_token_id} pad={tokenizer.pad_token_id}')
mx_len = 0

for filepath,dirnames,filenames in os.walk(test_text_path):
    for filename in filenames:
        with open(os.path.join(filepath,filename), 'rb') as f:
            raw_text = pickle.load(f)[0]['text']
            text = []
            for sent in raw_text:
                encoded_sent = tokenizer.encode(sent, return_tensors='pt', padding=True)
                mx_len = max(mx_len, len(encoded_sent))
                text.append(encoded_sent)
                print(encoded_sent.shape, model(encoded_sent)['last_hidden_state'].shape)
                exit(0)
            text_list.append(text)
print(f"max length : {mx_len}")

for filepath,dirnames,filenames in os.walk(test_motion_path):
    for filename in filenames:
        with open(os.path.join(filepath,filename), 'rb') as f:
            motion = pickle.load(f)
            motion_list.append(motion)

