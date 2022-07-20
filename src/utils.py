# Written by Seonwoo Min, LG AI Research (seonwoo.min0@gmail.com)

import os
import sys
import random
import hashlib
import datetime
import numpy as np

import torch


def Print(string, output, newline=False, timestamp=True):
    """ print to stdout and a file (if given) """
    if timestamp:
        time = datetime.datetime.now()
        line = '\t'.join([str(time.strftime('%m-%d %H:%M:%S')), string])
    else:
        time = None
        line = string

    if not output == sys.stdout:
        print(line, file=output)
        if newline: print("", file=output)
    else:
        print(line, file=sys.stderr)
        if newline: print("", file=sys.stderr)

    output.flush()
    return time


def seed_hash(*args):
    """ derive an integer hash from all args, for use as a random seed """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


def set_seeds(algorithm, test_env, seed):
    """ set random seeds """
    seed = seed_hash(algorithm, test_env, seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_output(args, string):
    """ set output configurations """
    output, save_prefix = sys.stdout, None
    if args["output_path"] is not None:
        save_prefix = args["output_path"]
        if not os.path.exists(save_prefix):
            os.makedirs(save_prefix, exist_ok=True)
        output = open(os.path.join(args["output_path"], "%s.txt" % string), "a")

    return output, save_prefix
