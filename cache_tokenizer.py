import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# tokenizer.save_pretrained("/public/home/CS172/tengzhh2022-cs172/finalproject/tokenizer/gpt2")

tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
model = AutoModel.from_pretrained('facebook/contriever-msmarco')


tokenizer.save_pretrained('/public/home/CS172/tengzhh2022-cs172/finalproject/tokenizer/contriever-msmarco')
model.save_pretrained('/public/home/CS172/tengzhh2022-cs172/finalproject/model/contriever-msmarco')


# tokenizer = AutoTokenizer.from_pretrained("/public/home/CS172/tengzhh2022-cs172/finalproject/tokenizer/gpt2")
# print(tokenizer.encode("I love KFC."))s