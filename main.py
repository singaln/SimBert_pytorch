#!/usr/bin/python3
# -*- coding : utf-8 -*-
# @Author : SinGaln
# @Time :  2021/12/8 16:16

import argparse
from trainer import Trainer
from tokenization import load_chinese_base_vocab, Tokenizer


def main(args):
    word2idx = load_chinese_base_vocab(args.vocab_path, simplfied=False)
    tokenizer = Tokenizer(word2idx)
    trainer = Trainer(args, tokenizer, args.vocab_size, word2idx, args.out_max_length)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="./simtrain_to05sts.txt", type=str, help="The train dataset path.")
    parser.add_argument("--vocab_path", default="./chinese_wwm_ext_pytorch/vocab.txt", type=str, help="vocab path.")
    parser.add_argument("--vocab_size", default=21128, type=str, help="The path for train data.")
    parser.add_argument("--out_max_length", default=12, type=str, help="label file path.")
    parser.add_argument("--vocab_file", default="./chinese_wwm_ext_pytorch/vocab.txt", type=str, help="Save path of new model.")
    parser.add_argument("--pretrain_model_path", default="./chinese_wwm_ext_pytorch", type=str,
                        help="Pretrained model path.")
    parser.add_argument("--learning_rate", default=1e-3, type=int, help="The seed of random.")
    parser.add_argument("--batch_size", default=32, type=int, help="The max sequence length of data.")
    parser.add_argument("--epoch", default=10, type=int,
                        help="Specifies a target value that is ignored and does not distribute to the input gradient.")
    parser.add_argument("--model_save_path", default="./simbert/", type=int, help="Embedding size of input data.")
    args = parser.parse_args()
    main(args)
