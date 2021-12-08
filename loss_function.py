#!/usr/bin/python3
# -*- coding : utf-8 -*-
# @Author : SinGaln
# @Time :  2021/12/7 19:00

import torch
import torch.nn as nn


class LossFun:
    def __init__(self, vocab_size, device):
        super(LossFun, self).__init__()
        self.vocab_size = vocab_size
        self.device = device

    def compute_loss_of_seq2seq(self, predictions, labels, target_mask):
        predictions = predictions.view(-1, self.vocab_size)
        labels = labels.view(-1)
        target_mask = target_mask.view(-1).float()
        loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        return (loss(predictions, labels) * target_mask).sum() / target_mask.sum()  # 通过mask 取消 pad 和句子a部分预测的影响

    def compute_loss_of_similarity(self, y_pred):
        y_true = self.get_labels_of_similarity(y_pred)  # 构建标签
        y_true = y_true.to(self.device)
        norm_a = torch.nn.functional.normalize(y_pred, dim=-1, p=2)
        # y_pred = K.l2_normalize(y_pred, axis=1)  # 句向量归一化
        similarities = norm_a.matmul(norm_a.t())

        # similarities = K.dot(y_pred, K.transpose(y_pred))  # 相似度矩阵
        similarities = similarities - (torch.eye(y_pred.shape[0]) * 1e12).to(self.device)  # 排除对角线
        similarities = similarities * 30  # scale
        similarities = similarities
        loss_f = nn.CrossEntropyLoss()
        loss = loss_f(similarities, y_true)
        return loss

    def get_labels_of_similarity(self, y_pred):
        idxs = torch.arange(0, y_pred.shape[0])
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        labels = (idxs_1 == idxs_2).float().argmax(dim=-1).long()
        return labels

    def compute_loss(self, predictions, labels, target_mask):
        loss1 = self.compute_loss_of_seq2seq(predictions, labels, target_mask)
        loss2 = self.compute_loss_of_similarity(predictions[:, 0])  # 拿出cls向量
        return loss1 + loss2
