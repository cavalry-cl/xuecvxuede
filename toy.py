import pickle
import os
from transformers import AutoTokenizer, AutoModel
from transformer_fixed_encoder.dataloader import MTDataset
from torch.utils.data import DataLoader
from transformer_fixed_encoder.model import make_model


test_text_path = "/public/home/CS172/tengzhh2022-cs172/finalproject/data/test/text"
test_motion_path = "/public/home/CS172/tengzhh2022-cs172/finalproject/data/test/motion"

# tokenizer_path = "/public/home/CS172/tengzhh2022-cs172/finalproject/tokenizer/gpt2"
tokenizer_path = "/public/home/CS172/tengzhh2022-cs172/finalproject/tokenizer/contriever-msmarco"
model_path = "/public/home/CS172/tengzhh2022-cs172/finalproject/model/contriever-msmarco"



tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# sentences = [
#     "Where was Marie Curie born?",
#     "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
#     "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
# ]

# inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
# print(inputs)

encoder_model = AutoModel.from_pretrained(model_path)
encoder_model.eval()

TGT_VOCAB = 1027

model = make_model(encoder_model, TGT_VOCAB)

ds = MTDataset(test_text_path, test_motion_path, tokenizer)
train_dataloader = DataLoader(ds, shuffle=True, batch_size=4, collate_fn=ds.collate_fn)
for d in train_dataloader:
    # print(d)
    print(d["src"].shape)
    print(d["tgt"].shape)
    print(d["src_mask"].shape)
    print(d["tgt_mask"].shape)
    print(encoder_model(input_ids=d["src"],attention_mask=d["src_mask"])["last_hidden_state"].shape)
    print(model(**d).shape)
    break
