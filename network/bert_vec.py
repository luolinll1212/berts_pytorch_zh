# -*- coding: utf-8 -*-
import numpy as np
import torch
from transformers import BertModel, BertConfig, BertTokenizer
from scipy.spatial.distance import cosine

from albert.albert_total import get_albert_total
from torch import nn


from conf.config import bert_config_path, bert_model_path, bert_vocab_path # bert_base_chinese
from conf.config import albert_config_path, albert_model_path, albert_vocab_path # albert_base


class BertTextNet(nn.Module):
    def __init__(self):
        """
        bert模型。
        """
        super(BertTextNet, self).__init__()
        modelConfig = BertConfig.from_pretrained(bert_config_path)
        self.textExtractor = BertModel.from_pretrained(
            bert_model_path, config=modelConfig)
        self.tokenizer = BertTokenizer.from_pretrained(bert_vocab_path)

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        return text_embeddings


class BertSeqVec(object):
    def __init__(self, text_net):
        """
        接收一个bert或albert模型，对文本进行向量化。
        :param text_net: bert或albert模型实例。
        """
        self.text_net = text_net
        self.tokenizer = text_net.tokenizer

    def seq2vec(self, text):
        """
        对文本向量化。
        :param text:str，未分词的文本。
        :return:
        """
        text = "[CLS] {} [SEP]".format(text)
        tokens, segments, input_masks = [], [], []

        tokenized_text = self.tokenizer.tokenize(text)  # 用tokenizer对句子分词
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表
        tokens.append(indexed_tokens)
        segments.append([0] * len(indexed_tokens))
        input_masks.append([1] * len(indexed_tokens))

        max_len = max([len(single) for single in tokens])  # 最大的句子长度

        for j in range(len(tokens)):
            padding = [0] * (max_len - len(tokens[j]))
            tokens[j] += padding
            segments[j] += padding
            input_masks[j] += padding
        tokens_tensor = torch.tensor(tokens)
        segments_tensors = torch.tensor(segments)
        input_masks_tensors = torch.tensor(input_masks)
        text_hashCodes = self.text_net(tokens_tensor, segments_tensors,
                                       input_masks_tensors)  # text_hashCodes是bert模型的文本特征
        return text_hashCodes[0].detach().numpy()


class AlbertTextNet(BertTextNet):
    def __init__(self):
        """
        albert 文本模型。
        """
        super(AlbertTextNet, self).__init__()
        config, tokenizer, model = get_albert_total(albert_config_path, albert_model_path, albert_vocab_path)
        self.textExtractor = model
        self.tokenizer = tokenizer

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        return text_embeddings


if __name__ == '__main__':
    # 模型初始化
    model = BertTextNet()  # 选择一个文本向量化模型
    seq2vec = BertSeqVec(model)  # 将模型实例给向量化对象。

    sentences = ["逍遥派掌门人无崖子为寻找一个色艺双全、聪明伶俐的徒弟，设下“珍珑”棋局，为少林寺虚字辈弟子虚竹误撞解开。",
                 "慕容复为应召拒绝王语嫣的爱情；众人救起伤心自杀的王语嫣，后段誉终于获得她的芳心。",
                 "鸠摩智贪练少林武功，走火入魔，幸被段誉吸去全身功力，保住性命，大彻大悟，成为一代高僧。",
                 "张无忌历尽艰辛，备受误解，化解恩仇，最终也查明了丐帮史火龙之死乃是成昆、陈友谅师徒所为",
                 "武氏与柯镇恶带着垂死的陆氏夫妇和几个小孩相聚，不料李莫愁尾随追来，打伤武三通",
                 "人工智能亦称智械、机器智能，指由人制造出来的机器所表现出来的智能。",
                 "人工智能的研究是高度技术性和专业的，各分支领域都是深入且各不相通的，因而涉及范围极广。",
                 "自然语言认知和理解是让计算机把输入的语言变成有意思的符号和关系，然后根据目的再处理。"]

    print(sentences)
    distances = []

    # 语料库向量化
    vec_sentences = []
    for sentence in sentences:
        vec = seq2vec.seq2vec(sentence)  # 向量化
        vec_sentences += [(vec)]
    # numpy类型
    np_vec_sentences = np.array(vec_sentences)
    print(np_vec_sentences.shape)
    # exit()

    np_test_vec = np.array([seq2vec.seq2vec("人工智能与自然语言处理")])
    print(np_test_vec.shape)
    # exit()

    score = np.sum(np_test_vec * np_vec_sentences, axis=1) / np.linalg.norm(np_vec_sentences, axis=1) / np.linalg.norm(np_test_vec, axis=1)
    print(score)

    index = np.argsort(score)[::-1][:1][0]
    print(index)
    cond = score[index]

    print(sentences[index])