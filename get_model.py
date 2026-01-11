import os
import torch
from transformers import AutoTokenizer, AutoModel
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
# model_id = "microsoft/mdeberta-v3-base"
# save_dir = "./models/mdeberta-v3-base"
# model_id = "microsoft/Multilingual-MiniLM-L12-H384"
# save_dir = "./models/Multilingual-MiniLM-L12-H384"

# os.makedirs(save_dir, exist_ok=True)

# # 1) 下载 tokenizer + model（第一次会联网下载）
# tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
# model = AutoModel.from_pretrained(model_id, torch_dtype="auto")

# # 2) 保存到本地目录
# tokenizer.save_pretrained(save_dir)
# model.save_pretrained(save_dir)

# print("Saved to:", save_dir)
