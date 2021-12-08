#!/usr/bin/python3
# -*- coding : utf-8 -*-
# @Author : SinGaln
# @Time :  2021/12/7 16:41

"""
主要思想：利用Unilm的mask方式进行训练
"""
from abc import ABC

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from loss_function import LossFun
from transformers.activations import ACT2FN
from transformers import BertModel, BertPreTrainedModel


class BertLayerNorm(nn.Module, ABC):
    """
    output = (gamma * (tensor - mean) / (std + eps)) + beta
    """

    def __init__(self, hidden_size, eps=1e-12, conditional=False):
        """
        Parameters initialization.
        """
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.conditional = conditional
        if conditional:
            # 说明是条件 ln
            self.weight_dense = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.weight_dense.weight.data.uniform_(0, 0)
            self.bias_dense = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.bias_dense.weight.data.uniform_(0, 0)

    def forward(self, x):
        if not self.conditional:
            # 针对最后一个维度求解
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.gamma * x + self.beta
        else:
            inputs = x[0]
            cond = x[1]
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(dim=1)

            weight = self.gamma + self.weight_dense(cond)
            bias = self.beta + self.bias_dense(cond)
            u = inputs.mean(-1, keepdim=True)
            s = (inputs - u).pow(2).mean(-1, keepdim=True)
            x = (inputs - u) / torch.sqrt(s + self.variance_epsilon)

            return weight * x + bias


class BertPredictionHeadTransform(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class Predictions(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, x):
        x = self.transform(x)

        return x, self.decoder(x)


class CLS(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.predictions = Predictions(config)

    def forward(self, x):
        return self.predictions(x)


class UniLM_Mask(object):
    """定义UniLM的Attention Mask（Seq2Seq模型用）
    其中source和target由segment_ids来表示。
    UniLM: https://arxiv.org/abs/1905.03197
    """

    def __init__(self, inputs, token_type_ids):
        """
        :param inputs: input_ids
        :param token_type_ids: token_type_ids
        """
        self.inputs = inputs
        self.token_type_ids = token_type_ids

    def mask_extend(self):
        forward_mask = self.token_type_ids.unsqueeze(1).unsqueeze(2).float()
        backward_mask = self.token_type_ids.unsqueeze(1).unsqueeze(3).float()
        return forward_mask, backward_mask

    def unilm_attention_mask(self):
        seq_length = self.inputs[1]
        attention_mask = torch.tril(torch.ones(seq_length, seq_length), diagonal=0)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        forward, backward = self.mask_extend()
        attention_mask = (1 - forward) * (1 - backward) + backward * attention_mask
        return attention_mask


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
        :param logits: logits distribution shape (vocabulary size)
        :param filter_value:
        :param top_p: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
        :param top_k: keep only top k tokens with highest probability (top-k filtering).
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


class SimBert(BertPreTrainedModel):
    def __init__(self, config, args, device, tokenizer, vocab_size, word2idx, out_max_length):
        super(SimBert, self).__init__(config)
        self.args = args
        self.device = device
        self.vocab_size = vocab_size
        self.simbert = BertModel(config=config)
        self.cls = CLS(config=config)
        self.word2idx = word2idx
        self.tokenizer = tokenizer
        self.out_max_length = out_max_length

    def forward(self, input_tensor, token_type_id, position_enc=None, labels=None):
        # 传入输入，位置编码，token type id ，还有句子a 和句子b的长度，注意都是传入一个batch数据
        #  传入的几个值，在seq2seq 的batch iter 函数里面都可以返回
        input_tensor = input_tensor.to(self.device)
        token_type_id = token_type_id.to(self.device)
        if position_enc is not None:
            position_enc = position_enc.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        attention_mask = UniLM_Mask(input_tensor, token_type_id)

        enc_layers, _ = self.simbert(input_tensor, position_ids=position_enc, token_type_ids=token_type_id,
                                     attention_mask=attention_mask,
                                     output_all_encoded_layers=True)
        squence_out = enc_layers[-1]  # 取出来最后一层输出

        _, predictions = self.cls(squence_out)
        loss_fun = LossFun(vocab_size=self.vocab_size, device=self.device)

        if labels is not None:
            # 计算loss
            # 需要构建特殊的输出mask 才能计算正确的loss
            # 预测的值不用取最后sep符号的结果 因此是到-1
            predictions = predictions[:, :-1].contiguous()
            target_mask = token_type_id[:, 1:].contiguous()
            loss = loss_fun.compute_loss(predictions, labels, target_mask)
            return predictions, loss
        else:
            return predictions

    def generate(self, text, out_max_length=40, beam_size=1, max_length=256):
        # 对 一个 句子生成相应的结果
        # 通过输出最大长度得到输入的最大长度，这里问题不大，如果超过最大长度会进行截断
        input_max_length = max_length - out_max_length
        # print(text)
        try:
            token_ids, token_type_ids = self.tokenizer.encode(text, max_length=input_max_length)
        except:
            # 可能是transformer的tokenizer
            tokenizer_out = self.tokenizer.encode_plus(text, max_length=input_max_length, truncation=True)
            token_ids = tokenizer_out["input_ids"]
            token_type_ids = tokenizer_out["token_type_ids"]

        token_ids = torch.tensor(token_ids, device=self.device).view(1, -1)
        token_type_ids = torch.tensor(token_type_ids, device=self.device).view(1, -1)

        out_puts_ids = self.beam_search(token_ids, token_type_ids, self.word2idx, beam_size=beam_size,
                                        device=self.device)

        return self.tokenizer.decode(out_puts_ids.cpu().numpy())

    def sample_generate(self, text, out_max_length=40, top_k=30, top_p=0.0, max_length=256):
        input_max_length = max_length - out_max_length
        token_ids, token_type_ids = self.tokenizer.encode(text, max_length=input_max_length)

        token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long).view(1, -1)
        token_type_ids = torch.tensor(token_type_ids, device=self.device, dtype=torch.long).view(1, -1)
        device = self.device
        output_ids = []
        sep_id = self.word2idx["[SEP]"]
        with torch.no_grad():
            for step in range(out_max_length):
                scores = self.forward(token_ids, token_type_ids)
                logit_score = torch.log_softmax(scores[:, -1], dim=-1).squeeze(0)
                logit_score[self.word2idx["[UNK]"]] = -float('Inf')
                filtered_logits = top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if sep_id == next_token.item():
                    break
                output_ids.append(next_token.item())
                token_ids = torch.cat((token_ids, next_token.long().unsqueeze(0)), dim=1)
                token_type_ids = torch.cat([token_type_ids, torch.ones((1, 1), device=device, dtype=torch.long)], dim=1)

        return self.tokenizer.decode(np.array(output_ids))

    def beam_search(self, token_ids, token_type_ids, word2idx, beam_size=1, device="cpu"):
        """
        beam-search操作
        """
        sep_id = word2idx["[SEP]"]

        # 用来保存输出序列
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        # 用来保存累计得分

        with torch.no_grad():
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            for step in range(self.out_max_length):
                if step == 0:
                    scores = self.forward(token_ids, token_type_ids)
                    # 重复beam-size次 输入ids
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                else:
                    scores = self.forward(new_input_ids, new_token_type_ids)

                logit_score = torch.log_softmax(scores[:, -1], dim=-1)

                logit_score = output_scores.view(-1, 1) + logit_score  # 累计得分
                # 取topk的时候我们是展平了然后再去调用topk函数
                # 展平
                logit_score = logit_score.view(-1)
                hype_score, hype_pos = torch.topk(logit_score, beam_size)
                indice1 = (hype_pos // scores.shape[-1])  # 行索引
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1)  # 列索引

                # 更新得分
                output_scores = hype_score
                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)

                end_counts = (output_ids == sep_id).sum(1)  # 统计出现的end标记
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    # 说明出现终止了～
                    return output_ids[best_one][:-1]
                else:
                    # 保留未完成部分
                    flag = (end_counts < 1)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        beam_size = flag.sum()  # topk相应变化

            return output_ids[output_scores.argmax()]
