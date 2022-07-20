# Written by Seonwoo Min, LG AI Research (seonwoo.min0@gmail.com)
# Some parts of the code ewre referenced from or inspired by below
# - DomainBed (github.com/facebookresearch/DomainBed)
# - GVE (htps://github.com/salaniz/pytorch-gve-lrcn)
# - MixStyle (https://github.com/KaiyangZhou/mixstyle-release/blob/master/imcls/models/resnet_mixstyle2.py)

import random

import torch
import torch.nn as nn
import torchvision.models
import torch.utils.model_zoo as model_zoo
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ResNet(torch.nn.Module):
    """ ResNet with the softmax chopped off and the batchnorm frozen """
    def __init__(self, num_channels=3, mixstyle=None):
        super(ResNet, self).__init__()
        if not mixstyle:
            resnet50 = torchvision.models.resnet50(pretrained=True)
        else:
            resnet50 = Resnet50_mixstyle(pretrained=True)
            pretrain_dict = model_zoo.load_url("https://download.pytorch.org/models/resnet50-19c8e357.pth")
            resnet50.load_state_dict(pretrain_dict, strict=False)

        if num_channels != 3:
            tmp = resnet50.conv1.weight.data.clone()
            resnet50.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            for i in range(3):
                resnet50.conv1.weight.data[:, i, :, :] = tmp[:, i, :, :]

        self.network = resnet50
        del self.network.fc
        self.network.fc = Identity()

        self.n_outputs = 2048
        self.freeze_bn()

    def forward(self, x):
        """ encode x into a feature vewor """
        x = self.network(x).reshape(len(x), -1)

        return x

    def train(self, mode=True):
        """ override the default train() to freeze the BN parameters """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class Explainer(torch.nn.Module):
    """ Textual Explanation Generator """
    def __init__(self, num_classes, vocab, proj_size, lstm_size):
        super(Explainer, self).__init__()
        self.embed = nn.Embedding(len(vocab), proj_size)
        self.lstm1 = nn.LSTM(proj_size, lstm_size, batch_first=True)
        self.lstm2 = nn.LSTM(proj_size + lstm_size + num_classes, lstm_size, batch_first=True)
        self.linear = nn.Linear(lstm_size, len(vocab))

        self.vocab = vocab
        self.start_word = torch.tensor([vocab(vocab.start_token)], dtype=torch.long)
        self.end_word = torch.tensor([vocab(vocab.end_token)], dtype=torch.long)

    def forward(self, x, y_hat, w, l):
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        l = l.cpu()

        w = self.embed(w)
        w = pack_padded_sequence(w, l, batch_first=True, enforce_sorted=False)
        w, _ = self.lstm1(w)
        w, _ = pad_packed_sequence(w, batch_first=True)

        x = x.unsqueeze(1).expand(-1, w.size(1), -1)
        y_hat = y_hat.unsqueeze(1).expand(-1, w.size(1), -1)
        w = torch.cat((w, x, y_hat), 2)
        w = pack_padded_sequence(w, l, batch_first=True, enforce_sorted=False)
        w, _ = self.lstm2(w)
        w, _ = pad_packed_sequence(w, batch_first=True)
        w = self.linear(w)

        return w

    def sample(self, x, y_hat, max_length=80):
        x = x.unsqueeze(1)
        y_hat = y_hat.unsqueeze(1)

        w = self.embed(self.start_word.to(x.device)).unsqueeze(0)
        w = w.expand(x.size(0), -1, -1)

        end_word = self.end_word.to(x.device).squeeze().expand(x.size(0))
        reached_end = torch.zeros_like(end_word.data).bool()
        lengths = torch.zeros_like(end_word.data).long()

        sampled_ids, log_ps = [], []
        states, states1, states2 = [], None, None
        for _ in range(max_length):
            if reached_end.all(): break
            output, states1 = self.lstm1(w, states1)
            output = torch.cat((output, x, y_hat), 2)
            output, states2 = self.lstm2(output, states2)
            output = self.linear(output.squeeze(1))

            prob = Categorical(logits=output)
            sampled_id = prob.sample()

            lengths += (~reached_end).long()
            sampled_ids.append(sampled_id)
            log_ps.append(prob.log_prob(sampled_id) * (~reached_end).float())
            states.append(states1[0].squeeze(0))

            reached_end = reached_end | sampled_id.eq(end_word).data
            w = self.embed(sampled_id).unsqueeze(1)

        sampled_ids = torch.stack(sampled_ids, 1)
        log_ps = torch.stack(log_ps, 1)
        
        states = torch.stack(states, 1)
        last_idxs = (lengths - 1).view(-1, 1, 1).expand(-1, -1, states.size(2))
        states = states.gather(1, last_idxs).squeeze(1)

        return sampled_ids, log_ps, states, lengths

    def generate(self, x, y_hat, max_length=80):
        x = x.unsqueeze(1)
        y_hat = y_hat.unsqueeze(1)

        w = self.embed(self.start_word.to(x.device)).unsqueeze(0)
        w = w.expand(x.size(0), -1, -1)

        end_word = self.end_word.to(x.device).squeeze().expand(x.size(0))
        reached_end = torch.zeros_like(end_word.data).bool()

        sampled_ids= []
        states1, states2 = None, None
        for _ in range(max_length):
            if reached_end.all(): break
            output, states1 = self.lstm1(w, states1)
            output = torch.cat((output, x, y_hat), 2)
            output, states2 = self.lstm2(output, states2)
            output = self.linear(output.squeeze(1))

            _, sampled_id = output.max(1)
            sampled_ids.append(sampled_id)
  
            reached_end = reached_end | sampled_id.eq(end_word).data
            w = self.embed(sampled_id).unsqueeze(1)

        sampled_ids = torch.stack(sampled_ids, 1)
        explanations = []
        for i in range(len(sampled_ids)):
            explanation = []
            for sampled_id in sampled_ids[i]:
                w = self.vocab.get_word_from_idx(sampled_id.data.item())
                if w == self.vocab.end_token: 
                    break
                elif w != self.vocab.start_token: 
                    explanation.append(w)   
            explanations.append(' '.join(explanation))

        return explanations


class Discriminator(torch.nn.Module):
    """ Sentence classifier to produce higher reward for class discriminative explanations """
    def __init__(self, num_classes, vocab, proj_size, lstm_size):
        super(Discriminator, self).__init__()
        self.embed = nn.Embedding(len(vocab), proj_size)
        self.lstm = nn.LSTM(proj_size, lstm_size, batch_first=True)
        self.linear = nn.Linear(lstm_size, num_classes)

    def forward(self, w, l):
        self.lstm.flatten_parameters()
        l = l.cpu()

        w = self.embed(w)
        w = pack_padded_sequence(w, l, batch_first=True, enforce_sorted=False)
        w, _ = self.lstm(w)
        w, _ = pad_packed_sequence(w, batch_first=True)

        last_idxs = (l - 1).view(-1, 1, 1).expand(-1, -1, w.size(2))
        w = w.gather(1, last_idxs.to(w.device)).squeeze()
        w = self.linear(w)

        return w


class ContextNet(nn.Module):
    """ ContextNet for ARM algorithm """
    def __init__(self):
        super(ContextNet, self).__init__()
        self.context_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.context_net(x)


class MLP(nn.Module):
    """ MLP for DANN/CDANN algorithms """
    def __init__(self, n_inputs, n_outputs):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_outputs)
        )

    def forward(self, x):
        return self.mlp(x)


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Resnet50_mixstyle(nn.Module):
    def __init__(self, pretrained=True):
        self.inplanes = 64
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.mixstyle = MixStyle()
        self._out_features = 2048
        self.fc = nn.Identity()

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.mixstyle(x)

        x = self.layer2(x)
        x = self.mixstyle(x)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_avgpool(x)

        return x.view(x.size(0), -1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MixStyle(nn.Module):
    """ MixStyle (w/ domain prior) """
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x

        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        perm = torch.arange(B - 1, -1, -1)  # inverse index
        perm_b, perm_a = perm.chunk(2)
        perm_b = perm_b[torch.randperm(B // 2)]
        perm_a = perm_a[torch.randperm(B // 2)]
        perm = torch.cat([perm_b, perm_a], 0)
        # perm = torch.randperm(B)

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix
