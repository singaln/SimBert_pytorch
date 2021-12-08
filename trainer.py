#!/usr/bin/python3
# -*- coding : utf-8 -*-
# @Author : SinGaln
# @Time :  2021/12/8 16:01

import time
import torch
from tqdm import tqdm
from model import SimBert
from transformers import BertConfig
from torch.utils.data import DataLoader
from data_loader import read_corpus, BertDataset, collate_fn


class Trainer:
    def __init__(self, args, tokenizer, vocab_size, word2idx, out_max_length):
        # 判断是否有可用GPU
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.configs = BertConfig.from_pretrained(args.pretrain_model_path)
        self.model = SimBert.from_pretrained(args.pretrain_model_path, config=self.configs, args=args,
                                             device=self.device, tokenizer=tokenizer, vocab_size=vocab_size,
                                             word2idx=word2idx, out_max_length=out_max_length)
        # 加载预训练的模型参数～
        inputs, outputs = read_corpus(args)
        # 加载已经训练好的模型，继续训练

        # 将模型发送到计算设备(GPU或CPU)
        self.model.to(self.device)
        # 多GPU
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        # 声明需要优化的参数
        self.optim_parameters = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=args.learning_rate, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = BertDataset(inputs, outputs, args, tokenizer)
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    def train(self):
        # 一个epoch的训练
        self.model.train()
        self.iteration(self.args.epoch, dataloader=self.dataloader, train=True)

    def save(self, save_path):
        """
        保存模型
        """
        self.model.save_all_params(save_path)
        print("{} saved!".format(save_path))

    def iteration(self, epoch, dataloader, train=True):
        total_loss = 0
        start_time = time.time()  # 得到当前时间
        step = 0
        report_loss = 0
        for token_ids, token_type_ids, target_ids in tqdm(dataloader, position=0, leave=True):
            step += 1
            if step % 1000 == 0:
                self.model.eval()
                test_data = [
                    "他这个人没个正经的。",
                    "咱俩谁跟谁呀。"
                ]
                for text in test_data:
                    print(self.model.sample_generate(text))
                    print(self.model.sample_generate(text))
                    print(self.model.sample_generate(text))
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("loss is " + str(report_loss))
                report_loss = 0
                # self.eval(epoch)
                self.model.train()
            if step % 5000 == 0:
                self.save(self.args.model_save_path)

            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.model(token_ids, token_type_ids, labels=target_ids)
            report_loss += loss.item()
            # 反向传播
            if train:
                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                # 用获取的梯度更新模型参数
                self.optimizer.step()

            # 为计算当前epoch的平均loss
            total_loss += loss.item()

        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        print("epoch is " + str(epoch) + ". loss is " + str(total_loss) + ". spend time is " + str(spend_time))
        # 保存模型
        self.save(self.args.model_save_path)
