# Written by Seonwoo Min, LG AI Research (seonwoo.min0@gmail.com)

import os
import json
import numpy as np
from collections import OrderedDict

import torch

from src.data import evaluate_text
from src.utils import Print


class Trainer():
    """ train / eval helper class """
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

        # initialize logging parameters
        self.logger_train = Logger()
        self.logger_eval = Logger()

    def train(self, minibatches):
        # training of the model
        minibatches = set_device(minibatches, self.device)

        self.model.train()
        loss_dict = self.model.update(minibatches)

        # logging
        self.logger_train.loss_update(loss_dict)

    def evaluate(self, minibatch, name, save_flag=False):
        # evaluation of the model
        minibatch = set_device(minibatch, self.device)

        self.model.eval()
        with torch.no_grad():
            correct, total, result_dict = self.model.evaluate(minibatch, name, save_flag)

        self.logger_eval.acc_update({name: [correct, total]})
        self.logger_eval.result_update(result_dict)
        
    def save_result(self, save_prefix, checkpoint, data_path):
        # save full-evaluation result """
        self.logger_eval.aggregate()

        if save_prefix is None: return
        elif not os.path.exists(os.path.join(save_prefix, "result/")):
            os.makedirs(os.path.join(save_prefix, "result/"), exist_ok=True)

        for k, v in self.logger_eval.result_dict.items():
            if isinstance(v, np.ndarray):
                np.save(os.path.join(save_prefix, "result/%s_%s.npy" % (checkpoint, k)), v)

            elif k.endswith("file-names"):
                text_labels = {"images":[], "annotations":[]}
                for image_id, file_name in enumerate(v):
                    text_labels["images"].append({"id": image_id, "file_name": file_name})
                    lines = open(os.path.join(data_path, "texts/%s" % (file_name.replace(".jpg", ".txt")))).readlines()
                    lines = [line.strip().lower() for line in lines if len(line.strip().split()) > 5]
                    for line in lines:
                        text_labels["annotations"].append({"image_id": image_id, "file_name": file_name, "id": len(text_labels["annotations"]), "caption": line})
                
                with open(os.path.join(save_prefix, "result/%s_%s.json" % (checkpoint, k.replace("file-names", "text-labels"))), "w") as file:
                    json.dump(text_labels, file)

            elif k.endswith("text-outputs"):
                text_outputs = [{"image_id": image_id, "caption": text} for image_id, text in enumerate(v)]

                with open(os.path.join(save_prefix, "result/%s_%s.json" % (checkpoint, k)), "w") as file:
                    json.dump(text_outputs, file)

    def save_model(self, step, save_prefix):
        # save a state_dict to checkpoint """
        if save_prefix is None: return
        elif not os.path.exists(os.path.join(save_prefix, "checkpoints/")):
            os.makedirs(os.path.join(save_prefix, "checkpoints/"), exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(save_prefix, "checkpoints/%d.pt" % step))

    def load_model(self, checkpoint, output):
        # load a state_dict from checkpoint """
        if checkpoint is None: return
        Print('loading a model state_dict from the checkpoint', output)
        checkpoint = torch.load(checkpoint, map_location=self.device)
        state_dict = OrderedDict()
        for k, v in checkpoint.items():
            state_dict[k] = v
        self.model.load_state_dict(state_dict)

    def headline(self, idx, loss_names, eval_names, output):
        # get a headline for logging
        if idx == "step":
            headline = [idx] + loss_names + eval_names
        else:
            headline = [idx] + eval_names

        Print("\t".join(headline), output)

    def log(self, step, output, save_prefix=None, checkpoint=None):
        # logging
        self.logger_train.aggregate()
        self.logger_eval.aggregate()

        if save_prefix is None:
            log = ["%04d" % step] + self.logger_train.log + self.logger_eval.log
            Print("\t".join(log), output)

        else:
            log = [str(step)] + self.logger_eval.log
            Print("\t".join(log), output)

            log_dict = OrderedDict()
            for k, v in self.logger_eval.result_dict.items():
                if k.endswith("text-outputs"):
                    labels_path = os.path.join(save_prefix, "result/%s_%s.json" % (checkpoint, k.replace("outputs", "labels")))
                    outputs_path = os.path.join(save_prefix, "result/%s_%s.json" % (checkpoint, k))
                    eval_results = evaluate_text(labels_path, outputs_path)
                    for metric in eval_results.keys():
                        if metric not in log_dict:
                            log_dict[metric] = []
                        log_dict[metric].append("%.4f" % eval_results[metric])

            for metric, log in log_dict.items():
                Print("\t".join([metric] + log), output)

        self.log_reset()

    def log_reset(self):
        # reset logging parameters
        self.logger_train.reset()
        self.logger_eval.reset()


class Logger():
    """ Logger class """
    def __init__(self):
        self.loss_dict = OrderedDict()
        self.acc_dict = OrderedDict()
        self.result_dict = OrderedDict()
        self.log_dict = OrderedDict()
        self.log = []

    def loss_update(self, loss_dict):
        # update loss_dict for current minibatch
        for k, v in loss_dict.items():
            if k not in self.loss_dict:
                self.loss_dict[k] = []
            self.loss_dict[k].append(v.item() if isinstance(v, torch.Tensor) else v)

    def acc_update(self, acc_dict):
        # update acc_dict for current minibatch
        for k, v in acc_dict.items():
            if k not in self.acc_dict:
                self.acc_dict[k] = [0, 0]
            self.acc_dict[k][0] += v[0]
            self.acc_dict[k][1] += v[1]

    def result_update(self, result_dict):
        # update result_dict for current minibatch
        for k, v in result_dict.items():
            if k not in self.result_dict:
                self.result_dict[k] = []
            if isinstance(v, np.ndarray):
                self.result_dict[k].append(v)
            else:
                self.result_dict[k] += v

    def aggregate(self):
        # aggregate logger dicts
        if len(self.log) == 0:
            for k, v in self.loss_dict.items():
                loss = np.mean(v)
                self.log_dict[k] = loss
                self.log.append("%.4f" % loss)

            for k, v in self.acc_dict.items():
                acc = v[0] / v[1]
                self.log_dict[k] = acc
                self.log.append("%.4f" % acc)

        for k, v in self.result_dict.items():
            if isinstance(v, list) and isinstance(v[0], np.ndarray):
                self.result_dict[k] = np.concatenate(v, axis=0)

    def reset(self):
        # reset logger
        self.loss_dict = OrderedDict()
        self.acc_dict = OrderedDict()
        self.result_dict = OrderedDict()
        self.log_dict = OrderedDict()
        self.log = []


def set_device(batch, device):
    """ recursive function for setting device for batch """
    if isinstance(batch, tuple) or isinstance(batch, list):
        return [set_device(t, device) for t in batch]
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch
