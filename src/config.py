# Written by Seonwoo Min, LG AI Research (seonwoo.min0@gmail.com)

import os
import sys
import json

import torch

from src.utils import Print


class GVRT_Config():
    def __init__(self, ste_flag):
        """ Grounding Visual Representations with Texts (GVRT) algorithm configurations """
        self.align_loss = True
        self.expl_loss = True
        self.align_loss_lambda = 1.0
        self.expl_loss_lambda = 1.0
        self.ste_flag = ste_flag
        self.proj_size = 128
        self.lstm_size = 128
        self.embed_size = 512 if not self.ste_flag else self.lstm_size 
        

def print_configs(args, device, output):
    Print(" ".join(['##### arguments #####']), output)
    Print(" ".join(['algorithm:', str(args["algorithm"])]), output)
    if args["algorithm"] == "GVRT":
        Print(" ".join(['ste:', str(args["ste"])]), output)
    Print(" ".join(['test_env:', str(args["test_env"])]), output)
    Print(" ".join(['seed:', str(args["seed"])]), output)
    if "checkpoint" in args:
        Print(" ".join(['checkpoint:', str(args["checkpoint"])]), output)
    Print(" ".join(['output_path:', str(args["output_path"])]), output)
    Print(" ".join(['device: %s (%d GPUs)' % (device, torch.cuda.device_count())]), output)
