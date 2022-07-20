# Written by Seonwoo Min, LG AI Research (seonwoo.min0@gmail.com)

import os
import sys
import argparse
os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch

from src.config import GVRT_Config, print_configs
from src.data import get_datasets_and_iterators
from src.algorithms import get_algorithm_class
from src.train import Trainer
from src.utils import Print, set_seeds, set_output

parser = argparse.ArgumentParser('Train a Domain Generalization Model for the CUB-DG dataset')
parser.add_argument('--algorithm', help='Domain generalization algorithm')
parser.add_argument('--ste', default=False, action='store_true', help='GVRT with STE')
parser.add_argument('--test-env', type=int, help='test environment (used for multi-source DG)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--output-path', help='path for outputs (default: stdout and without saving)')


def main():
    args = vars(parser.parse_args())
    gvrt_flag, gvrt_config = args["algorithm"] == "GVRT", GVRT_Config(args["ste"])
    env_flag = args["test_env"]
    output, save_prefix = set_output(args, "train_model_log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_configs(args, device, output)
    set_seeds(args["algorithm"], env_flag, args["seed"])

    ## Loading datasets
    start = Print(" ".join(['start loading datasets']), output)
    datasets, iterators_train, iterators_eval, eval_names = get_datasets_and_iterators(env_flag, gvrt_flag, eval_flag=False)
    end = Print('end loading datasets', output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## setup trainer configurations
    start = Print('start setting trainer configurations', output)
    algorithm_class = get_algorithm_class(args["algorithm"])
    if gvrt_flag:
        model = algorithm_class(datasets[0].num_classes, datasets[0].vocab, gvrt_config)
    else:
        model = algorithm_class(datasets[0].num_classes)
    trainer = Trainer(model, device)
    end = Print('end setting trainer configurations', output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## train a model
    N_STEPS, CHECKPOINT_FREQ = 5000, 300
    start = Print('start training a model', output)
    trainer.headline("step", model.loss_names, eval_names, output)
    for step in range(N_STEPS):
        ### train
        minibatches = next(iterators_train)
        trainer.train(minibatches)
        
        if (step + 1) % 10 == 0:
            print('# step [{}/{}]'.format(step + 1, N_STEPS), end='\r', file=sys.stderr)
        if ((step + 1) % CHECKPOINT_FREQ == 0) or (step + 1 == N_STEPS):
            for iterator_eval, eval_name in zip(iterators_eval, eval_names):
                for B, minibatch in enumerate(iterator_eval):
                    trainer.evaluate(minibatch, eval_name)
                    if B % 5 == 0: print('# step [{}/{}] {} {:.1%}'.format(step + 1, N_STEPS, eval_name, B / len(iterator_eval)), end='\r', file=sys.stderr)
                print(' ' * 50, end='\r', file=sys.stderr)

            trainer.save_model(step + 1, save_prefix)
            trainer.log(step + 1, output)

    end = Print('end training a model', output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)
    if not output == sys.stdout: output.close()


if __name__ == '__main__':
    main()
