# Written by Seonwoo Min, LG AI Research (seonwoo.min0@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - DomainBed (github.com/facebookresearch/DomainBed)
# - GVE (htps://github.com/salaniz/pytorch-gve-lrcn)

import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.checkpoint import checkpoint

from src.networks import ResNet, Explainer, Discriminator, ContextNet, MLP


def get_algorithm_class(algorithm):
    """ Return the algorithm class with the given name """
    if algorithm not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm))
    return globals()[algorithm]


def get_optimizer(named_parameters, GVRT_flag=False, betas=(0.9, 0.999)):
    """ configure optim and scheduler """
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0

    params, params_extra = [], []
    for name, param in named_parameters:
        if not GVRT_flag or name.startswith("featurizer"):
            params.append(param)
        else:
            params_extra.append(param)

    optimizer = torch.optim.Adam([{"params": params}, {"params": params_extra, "lr": LEARNING_RATE * 10}],
                                 lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=betas)

    return optimizer


class ERM(torch.nn.Module):
    """ Empirical Risk Minimization (ERM) """
    def __init__(self, num_classes):
        super(ERM, self).__init__()
        self.featurizer = ResNet()
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)

        self.loss_names = ["loss", "task_loss"]
        self.optimizer = get_optimizer(self.named_parameters())

    def update(self, minibatches):
        num_domains = len(minibatches)

        loss_dict = OrderedDict({loss_name: 0 for loss_name in self.loss_names})
        for minibatch in minibatches:
            x, y = minibatch

            x = self.featurizer(x)
            y_hat = self.classifier(x)
            loss_dict["task_loss"] += F.cross_entropy(y_hat, y) / num_domains

        for loss_name in self.loss_names:
            if loss_name != "loss":
                loss_dict["loss"] += loss_dict[loss_name]

        self.optimizer.zero_grad()
        loss_dict["loss"].backward()
        self.optimizer.step()

        return loss_dict

    def evaluate(self, minibatch, name, save_flag):
        x, y = minibatch

        x = self.featurizer(x)
        y_hat = self.classifier(x)
        correct = (y_hat.argmax(1).eq(y).float()).sum().item()
        total = float(len(x))

        result_dict = OrderedDict()
        if save_flag:
            result_dict[name + "_task-features"] = x.cpu().detach().numpy()
            result_dict[name + "_task-outputs"] = y_hat.cpu().detach().numpy()
            result_dict[name + "_task-labels"] = y.cpu().detach().numpy()

        return correct, total, result_dict


class MixStyle(ERM):
    """MixStyle w/ domain label"""
    def __init__(self, num_classes):
        super(ERM, self).__init__()
        self.featurizer = ResNet(mixstyle=True)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)

        self.loss_names = ["loss", "mixstyle_loss"]
        self.optimizer = get_optimizer(self.named_parameters())

    def update(self, minibatches):
        num_domains = len(minibatches)

        loss_dict = OrderedDict({loss_name: 0 for loss_name in self.loss_names})
        for (xi, yi), (xj, yj) in self.random_pairs_of_minibatches(minibatches):
            x = torch.cat([xi, xj])
            y = torch.cat([yi, yj])

            x = self.featurizer(x)
            y_hat = self.classifier(x)

            loss_dict["mixstyle_loss"] += F.cross_entropy(y_hat, y) / num_domains

        for loss_name in self.loss_names:
            if loss_name != "loss":
                loss_dict["loss"] += loss_dict[loss_name]

        self.optimizer.zero_grad()
        loss_dict["loss"].backward()
        self.optimizer.step()

        return loss_dict

    def random_pairs_of_minibatches(self, minibatches):
        minibatches = [[x.chunk(2), y.chunk(2)] for x, y in minibatches]
        perm = torch.randperm(len(minibatches)).tolist()

        pairs = []
        for i in range(len(minibatches)):
            j = i + 1 if i < (len(minibatches) - 1) else 0
            xi, yi = minibatches[perm[i]][0][0], minibatches[perm[i]][1][0]
            xj, yj = minibatches[perm[j]][0][1], minibatches[perm[j]][1][1]
            min_n = min(len(xi), len(xj))
            pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

        return pairs


class ARM(ERM):
    """ Adaptive Risk Minimization (ARM) """
    def __init__(self, num_classes):
        super(ARM, self).__init__(num_classes)
        self.featurizer = ResNet(num_channels=4)
        self.context_net = ContextNet()

        self.optimizer = get_optimizer(self.named_parameters())

    def update(self, minibatches):
        num_domains = len(minibatches)

        loss_dict = OrderedDict({loss_name: 0 for loss_name in self.loss_names})
        for minibatch in minibatches:
            x, y = minibatch

            x.requires_grad = True
            context = checkpoint(self.context_net, x)
            context = context.mean(dim=0, keepdim=True)
            context = torch.repeat_interleave(context, repeats=len(x), dim=0)
            x = torch.cat([x, context], dim=1)

            x = self.featurizer(x)
            y_hat = self.classifier(x)
            loss_dict["task_loss"] += F.cross_entropy(y_hat, y) / num_domains

        for loss_name in self.loss_names:
            if loss_name != "loss":
                loss_dict["loss"] += loss_dict[loss_name]

        self.optimizer.zero_grad()
        loss_dict["loss"].backward()
        self.optimizer.step()

        return loss_dict

    def evaluate(self, minibatch, name, save_flag):
        x, y = minibatch

        x.requires_grad = True
        context = checkpoint(self.context_net, x)
        context = context.mean(dim=0, keepdim=True)
        context = torch.repeat_interleave(context, repeats=len(x), dim=0)
        x = torch.cat([x, context], dim=1)

        x = self.featurizer(x)
        y_hat = self.classifier(x)
        correct = (y_hat.argmax(1).eq(y).float()).sum().item()
        total = float(len(x))

        result_dict = OrderedDict()
        if save_flag:
            result_dict[name + "_task-features"] = x.cpu().detach().numpy()
            result_dict[name + "_task-outputs"] = y_hat.cpu().detach().numpy()
            result_dict[name + "_task-labels"] = y.cpu().detach().numpy()

        return correct, total, result_dict


class AbstractDANN(ERM):
    """ Domain-Adversarial Neural Networks (abstract class) """
    def __init__(self, num_classes, conditional):
        super(AbstractDANN, self).__init__(num_classes)
        self.discriminator = MLP(self.featurizer.n_outputs, 4)
        self.conditional = conditional
        if self.conditional:
            self.class_embeddings = nn.Embedding(num_classes, self.featurizer.n_outputs)

        self.loss_names = ['loss', 'task_loss', 'disc_loss']
        self.optimizer_g = get_optimizer(list(self.featurizer.named_parameters()) +
                                         list(self.classifier.named_parameters()), betas=(0.5, 0.9))
        parameters_d = list(self.discriminator.named_parameters())
        if self.conditional:
            parameters_d += list(self.class_embeddings.named_parameters())
        self.optimizer_d = get_optimizer(parameters_d, betas=(0.5, 0.9))

    def update(self, minibatches):
        num_domains = len(minibatches)

        loss_dict = OrderedDict({loss_name: 0 for loss_name in self.loss_names})
        for m, minibatch in enumerate(minibatches):
            x, y = minibatch

            x = self.featurizer(x)

            if self.conditional:
                disc_out = self.discriminator(x + self.class_embeddings(y))
                disc_labels = torch.full_like(y, m)
                y_counts = F.one_hot(y).sum(dim=0)
                weights = 1. / (y_counts[y] * y_counts.shape[0]).float()
                loss_dict["disc_loss"] += (weights * F.cross_entropy(disc_out, disc_labels, reduction='none')).sum() / num_domains
            else:
                disc_out = self.discriminator(x)
                disc_labels = torch.full_like(y, m)
                loss_dict["disc_loss"] += F.cross_entropy(disc_out, disc_labels) / num_domains

        self.optimizer_d.zero_grad()
        loss_dict["disc_loss"].backward()
        self.optimizer_d.step()


        loss_dict = OrderedDict({loss_name: 0 for loss_name in self.loss_names})
        for m, minibatch in enumerate(minibatches):
            x, y = minibatch

            x = self.featurizer(x)
            y_hat = self.classifier(x)
            loss_dict["task_loss"] += F.cross_entropy(y_hat, y) / num_domains

            if self.conditional:
                disc_out = self.discriminator(x + self.class_embeddings(y))
                disc_labels = torch.full_like(y, m)
                y_counts = F.one_hot(y).sum(dim=0)
                weights = 1. / (y_counts[y] * y_counts.shape[0]).float()
                loss_dict["disc_loss"] += (weights * F.cross_entropy(disc_out, disc_labels, reduction='none')).sum() / num_domains
            else:
                disc_out = self.discriminator(x)
                disc_labels = torch.full_like(y, m)
                loss_dict["disc_loss"] += F.cross_entropy(disc_out, disc_labels) / num_domains

        self.optimizer_g.zero_grad()
        (loss_dict["task_loss"] - 0.1 * loss_dict["disc_loss"]).backward()
        self.optimizer_g.step()

        for loss_name in self.loss_names:
            if loss_name != "loss":
                loss_dict["loss"] += loss_dict[loss_name]

        return loss_dict


class DANN(AbstractDANN):
    """ Unconditional DANN """
    def __init__(self, num_classes):
        super(DANN, self).__init__(num_classes, conditional=False)


class CDANN(AbstractDANN):
    """ Conditional DANN """
    def __init__(self, num_classes):
        super(CDANN, self).__init__(num_classes, conditional=True)


class IRM(ERM):
    """ Invariant Risk Minimization """
    def __init__(self, num_classes):
        super(IRM, self).__init__(num_classes)
        self.loss_names = ["loss", "task_loss", "penalty_loss"]

    def update(self, minibatches):
        num_domains = len(minibatches)

        loss_dict = OrderedDict({loss_name: 0 for loss_name in self.loss_names})
        for minibatch in minibatches:
            x, y = minibatch

            x = self.featurizer(x)
            y_hat = self.classifier(x)
            loss_dict["task_loss"] += F.cross_entropy(y_hat, y) / num_domains
            loss_dict["penalty_loss"] += self.irm_penalty(y_hat, y) / num_domains

        for loss_name in self.loss_names:
            if loss_name != "loss":
                loss_dict["loss"] += loss_dict[loss_name]

        self.optimizer.zero_grad()
        loss_dict["loss"].backward()
        self.optimizer.step()

        return loss_dict

    def irm_penalty(self, y_hat, y):
        scale = torch.tensor(1.).to(y.device).requires_grad_()
        loss_1 = F.cross_entropy(y_hat[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(y_hat[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        penalty = torch.sum(grad_1 * grad_2)
        return penalty


class VREx(ERM):
    """ VREx from http://arxiv.org/abs/2003.00688 """
    def __init__(self, num_classes):
        super(VREx, self).__init__(num_classes)
        self.loss_names = ["loss", "task_loss", "penalty_loss"]

    def update(self, minibatches):
        num_domains = len(minibatches)
        losses = torch.zeros(len(minibatches)).to(minibatches[0][0].device)

        loss_dict = OrderedDict({loss_name: 0 for loss_name in self.loss_names})
        for m, minibatch in enumerate(minibatches):
            x, y = minibatch

            x = self.featurizer(x)
            y_hat = self.classifier(x)
            task_loss = F.cross_entropy(y_hat, y)
            loss_dict["task_loss"] += task_loss / num_domains
            losses[m] = task_loss

        loss_dict["penalty_loss"] += ((losses - loss_dict["task_loss"]) ** 2).mean()

        for loss_name in self.loss_names:
            if loss_name != "loss":
                loss_dict["loss"] += loss_dict[loss_name]

        self.optimizer.zero_grad()
        loss_dict["loss"].backward()
        self.optimizer.step()

        return loss_dict


class Mixup(ERM):
    """ Mixup of minibatches from different domains """
    def __init__(self, num_classes):
        super(Mixup, self).__init__(num_classes)
        self.loss_names = ["loss", "mixup_loss"]

    def update(self, minibatches):
        num_domains = len(minibatches)

        loss_dict = OrderedDict({loss_name: 0 for loss_name in self.loss_names})
        for (xi, yi), (xj, yj) in self.random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(0.2, 0.2)

            x = lam * xi + (1 - lam) * xj
            x = self.featurizer(x)
            y_hat = self.classifier(x)

            loss_dict["mixup_loss"] += lam * F.cross_entropy(y_hat, yi) / num_domains
            loss_dict["mixup_loss"] += (1 - lam) * F.cross_entropy(y_hat, yj) / num_domains

        for loss_name in self.loss_names:
            if loss_name != "loss":
                loss_dict["loss"] += loss_dict[loss_name]

        self.optimizer.zero_grad()
        loss_dict["loss"].backward()
        self.optimizer.step()

        return loss_dict

    def random_pairs_of_minibatches(self, minibatches):
        perm = torch.randperm(len(minibatches)).tolist()

        pairs = []
        for i in range(len(minibatches)):
            j = i + 1 if i < (len(minibatches) - 1) else 0
            xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
            xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]
            min_n = min(len(xi), len(xj))
            pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

        return pairs


class GroupDRO(ERM):
    """ Robust ERM minimizes the error at the worst minibatch """
    def __init__(self, num_classes):
        super(GroupDRO, self).__init__(num_classes)
        self.q = None

        self.loss_names = ["loss", "groupdro_loss"]

    def update(self, minibatches):
        num_domains = len(minibatches)
        if self.q is None:
            self.q = torch.ones(num_domains).to(minibatches[0][0].device)
        losses = torch.ones(num_domains).to(minibatches[0][0].device)

        loss_dict = OrderedDict({loss_name: 0 for loss_name in self.loss_names})
        for m, minibatch in enumerate(minibatches):
            x, y = minibatch

            x = self.featurizer(x)
            y_hat = self.classifier(x)
            losses[m] += F.cross_entropy(y_hat, y)
            self.q[m] *= (0.01 * losses[m].data).exp()

        self.q /= self.q.sum()
        loss_dict["groupdro_loss"] = torch.dot(losses, self.q)

        for loss_name in self.loss_names:
            if loss_name != "loss":
                loss_dict["loss"] += loss_dict[loss_name]

        self.optimizer.zero_grad()
        loss_dict["loss"].backward()
        self.optimizer.step()

        return loss_dict


class CORAL(ERM):
    """ ERM while matching the pair-wise domain feature distributions using mean and covariance difference """
    def __init__(self, num_classes):
        super(CORAL, self).__init__(num_classes)
        self.loss_names = ["loss", "task_loss", "mmd_loss"]

    def update(self, minibatches):
        num_domains = len(minibatches)
        num_domains_pair = (num_domains * (num_domains - 1) / 2)
        x_mb = []

        loss_dict = OrderedDict({loss_name: 0 for loss_name in self.loss_names})
        for minibatch in minibatches:
            x, y = minibatch

            x = self.featurizer(x)
            y_hat = self.classifier(x)
            loss_dict["task_loss"] += F.cross_entropy(y_hat, y) / num_domains
            x_mb.append(x)

        for i in range(num_domains):
            for j in range(i + 1, num_domains):
                loss_dict["mmd_loss"] += self.mmd(x_mb[i], x_mb[j]) / num_domains_pair

        for loss_name in self.loss_names:
            if loss_name != "loss":
                loss_dict["loss"] += loss_dict[loss_name]

        self.optimizer.zero_grad()
        loss_dict["loss"].backward()
        self.optimizer.step()

        return loss_dict

    def mmd(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff


class SagNet(ERM):
    """ Style Agnostic Network Algorithm 1 from: https://arxiv.org/abs/1910.11645 """
    def __init__(self, num_classes):
        super(SagNet, self).__init__(num_classes)
        self.classifier_s = nn.Linear(self.featurizer.n_outputs, num_classes)

        self.loss_names = ['loss', 'task_loss', 'style_loss', 'adv_loss']
        self.optimizer_f = get_optimizer(self.featurizer.named_parameters())
        self.optimizer_c = get_optimizer(self.classifier.named_parameters())
        self.optimizer_s = get_optimizer(self.classifier_s.named_parameters())

    def randomize(self, x, what="style", eps=1e-5):
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).to(x.device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches):
        num_domains = len(minibatches)

        loss_dict = OrderedDict({loss_name: 0 for loss_name in self.loss_names})
        for m, minibatch in enumerate(minibatches):
            x, y = minibatch

            x = self.featurizer(x)
            x = self.randomize(x, "style")
            y_hat = self.classifier(x)
            loss_dict["task_loss"] += F.cross_entropy(y_hat, y) / num_domains

        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_dict["task_loss"].backward()
        self.optimizer_f.step()
        self.optimizer_c.step()


        for m, minibatch in enumerate(minibatches):
            x, y = minibatch

            x = self.featurizer(x)
            x = self.randomize(x, "content")
            y_hat = self.classifier_s(x)
            loss_dict["style_loss"] += F.cross_entropy(y_hat, y) / num_domains

        self.optimizer_s.zero_grad()
        loss_dict["style_loss"].backward()
        self.optimizer_s.step()


        for m, minibatch in enumerate(minibatches):
            x, y = minibatch

            x = self.featurizer(x)
            x = self.randomize(x, "content")
            y_hat = self.classifier_s(x)
            loss_dict["adv_loss"] += 0.1 * -F.log_softmax(y_hat, dim=1).mean(1).mean() / num_domains

        self.optimizer_f.zero_grad()
        loss_dict["adv_loss"].backward()
        self.optimizer_f.step()

        for loss_name in self.loss_names:
            if loss_name != "loss":
                loss_dict["loss"] += loss_dict[loss_name]

        return loss_dict


class SD(ERM):
    """ Gradient starvation: a learning proclivity in neural networks """
    def __init__(self, num_classes):
        super(SD, self).__init__(num_classes)
        self.loss_names = ["loss", "task_loss", "sd_loss"]

    def update(self, minibatches):
        num_domains = len(minibatches)

        loss_dict = OrderedDict({loss_name: 0 for loss_name in self.loss_names})
        for minibatch in minibatches:
            x, y = minibatch

            x = self.featurizer(x)
            y_hat = self.classifier(x)
            loss_dict["task_loss"] += F.cross_entropy(y_hat, y) / num_domains
            loss_dict["sd_loss"] += 0.1 * (y_hat ** 2).mean() / num_domains

        for loss_name in self.loss_names:
            if loss_name != "loss":
                loss_dict["loss"] += loss_dict[loss_name]

        self.optimizer.zero_grad()
        loss_dict["loss"].backward()
        self.optimizer.step()

        return loss_dict


class GVRT(ERM):
    """ Grounding Visual Representations with Texts (GVRT) """
    def __init__(self, num_classes, vocab, GVRT_config):
        super(GVRT, self).__init__(num_classes)
        self.projector_v = nn.Linear(self.featurizer.n_outputs, GVRT_config.proj_size)
        self.projector_t = nn.Linear(GVRT_config.embed_size, GVRT_config.proj_size) if GVRT_config.align_loss else None
        self.classifier_p = nn.Linear(GVRT_config.proj_size, num_classes) if GVRT_config.align_loss else None
        self.explainer = Explainer(num_classes, vocab, GVRT_config.proj_size, GVRT_config.lstm_size) if GVRT_config.expl_loss else None
        self.discriminator = Discriminator(num_classes, vocab, GVRT_config.proj_size, GVRT_config.lstm_size) if GVRT_config.expl_loss else None

        self.loss_names = ["loss", "task_loss"]
        if GVRT_config.align_loss:
            self.loss_names += ["align_loss"]
        if GVRT_config.expl_loss:
            self.loss_names += ["expl_loss"]
        self.align_loss_lambda = GVRT_config.align_loss_lambda
        self.expl_loss_lambda = GVRT_config.expl_loss_lambda
        self.optimizer = get_optimizer(self.named_parameters(), GVRT_flag=True)
        self.ste_flag = GVRT_config.ste_flag

    def update(self, minibatches):
        num_domains = len(minibatches)

        loss_dict = OrderedDict({loss_name: 0 for loss_name in self.loss_names})
        for minibatch in minibatches:
            x, y, ts, tw, l, file_names = minibatch
            l = l.cpu()

            x = self.featurizer(x)
            y_hat = self.classifier(x)
            loss_dict["task_loss"] += F.cross_entropy(y_hat, y) / num_domains

            proj_x = self.projector_v(x)
            if "expl_loss" in loss_dict:
                expl_outputs = self.explainer(proj_x, F.one_hot(y, y_hat.size(1)), tw[:, :-1], l)
                expl_outputs = pack_padded_sequence(expl_outputs, l, batch_first=True, enforce_sorted=False)[0]
                targets = pack_padded_sequence(tw[:, 1:], l, batch_first=True, enforce_sorted=False)[0]

                sampled_t, log_ps, h, sampled_l = self.explainer.sample(proj_x, F.one_hot(y, y_hat.size(1)))
                with torch.no_grad():
                    disc_outputs = self.discriminator(sampled_t, sampled_l)
                    rewards = F.softmax(disc_outputs, dim=1).gather(1, y.view(-1, 1)).squeeze()
                disc_outputs = self.discriminator(tw[:, 1:], l)

                loss_dict["expl_loss"] += (self.expl_loss_lambda * F.cross_entropy(expl_outputs, targets) +
                                           self.expl_loss_lambda * -(log_ps.sum(dim=1) * rewards).sum() / len(y) +
                                           F.cross_entropy(disc_outputs, y)) / num_domains

            if "align_loss" in loss_dict:
                proj_t = self.projector_t(ts if not self.ste_flag else h)
                proj_outputs = self.classifier_p(proj_x)
                loss_dict["align_loss"] += (self.align_loss_lambda * F.mse_loss(proj_x, proj_t) +
                                            self.align_loss_lambda * F.cross_entropy(proj_outputs, y)) / num_domains

        for loss_name in self.loss_names:
            if loss_name != "loss":
                loss_dict["loss"] += loss_dict[loss_name]

        self.optimizer.zero_grad()
        loss_dict["loss"].backward()
        self.optimizer.step()

        return loss_dict

    def evaluate(self, minibatch, name, save_flag):
        x, y, ts, tw, l, file_names = minibatch

        x = self.featurizer(x)
        y_hat = self.classifier(x)
        correct = (y_hat.argmax(1).eq(y).float()).sum().item()
        total = float(len(x))

        result_dict = OrderedDict()
        if save_flag:
            result_dict[name + "_task-features"] = x.cpu().detach().numpy()
            result_dict[name + "_task-outputs"] = y_hat.cpu().detach().numpy()
            result_dict[name + "_task-labels"] = y.cpu().detach().numpy()
            result_dict[name + "_file-names"] = file_names

            if self.explainer is not None:
                proj_x = self.projector_v(x)
                captions = self.explainer.generate(proj_x, F.one_hot(torch.argmax(y_hat, 1), y_hat.size(1)))
                result_dict[name + "_text-outputs"] = captions

        return correct, total, result_dict
