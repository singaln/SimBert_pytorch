#!/usr/bin/python3
# -*- coding : utf-8 -*-
# @Author : SinGaln
# @Time :  2021/12/8 14:47

import torch
from torch.utils.data import Dataset

def read_corpus(args):
    with open(args.data_path, encoding="utf-8") as f:
        lines = f.readlines()
    inputs = []
    outputs = []
    for line in lines:
        line = line.split("\t")
        inputs.append(line[1])
        outputs.append(line[3])
    return inputs, outputs


class BertDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """

    def __init__(self, inputs, outputs, args, tokenizer):
        # 一般init函数是加载所有数据
        super(BertDataset, self).__init__()
        self.args = args
        self.inputs = inputs
        self.outputs = outputs
        self.tokenizer = tokenizer
        self.idx2word = {k: v for v, k in self.args.word2idx.items()}

    def __getitem__(self, i):
        # 得到单个数据
        # print(i)
        inp = self.inputs[i]
        out = self.outputs[i]
        token_ids_1, token_type_ids_1 = self.tokenizer.encode(
            inp, out, max_length=self.args.max_length
        )
        token_ids_2, token_type_ids_2 = self.tokenizer.encode(
            out, inp, max_length=self.args.max_length
        )

        output = {
            "token_ids_1": token_ids_1,
            "token_type_ids_1": token_type_ids_1,
            "token_ids_2": token_ids_2,
            "token_type_ids_2": token_type_ids_2,

        }
        return output

    def __len__(self):
        return len(self.inputs)


def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = []
    token_type_ids = []
    for data in batch:
        token_ids.append(data["token_ids_1"])
        token_type_ids.append(data["token_type_ids_1"])
        token_ids.append(data["token_ids_2"])
        token_type_ids.append(data["token_type_ids_2"])

    max_length = max([len(t) for t in token_ids])

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded
