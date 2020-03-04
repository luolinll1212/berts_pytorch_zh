# -*- coding: utf-8 -*-
import os

# bert pytorch的权重
path_bert_base_chinese = r"C:\Software_App\bert_zh_pytorch\bert\bert-base-chinese" # 点击 ReadMe bert-base-chinese
path_albert_base = r"C:\Software_App\bert_zh_pytorch\albert\albert_base" # 点击 ReadMe albert_base_zh

# 配置环境参数
# bert-base-chinese
bert_config_path = os.path.join(path_bert_base_chinese, "bert_config.json")
bert_model_path = os.path.join(path_bert_base_chinese, "pytorch_model.bin")
bert_vocab_path = os.path.join(path_bert_base_chinese, "vocab.txt")

# albert_base
albert_config_path = os.path.join(path_albert_base, "config.json")
albert_model_path = os.path.join(path_albert_base, "pytorch_model.bin")
albert_vocab_path = os.path.join(path_albert_base, "vocab.txt")

