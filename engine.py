# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import os
import time
from datetime import datetime

import math
# from flops_counter import get_model_complexity_info
import sys

from sklearn.manifold import TSNE
from thop import profile
import numpy as np
import seaborn as sns
from typing import Iterable, Optional

import torch

import timm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from torch import nn

from losses import DistillationLoss
import utils



class cls_head(nn.Module):
    def __init__(self):
        super(cls_head, self).__init__()
        self.num_features = 64
        self.head = nn.Linear(self.num_features, 12)
    def forward(self, x):
        x = self.head(x)
        return x

def plot_embedding_2D(data, label, title, datasetName='unimib', color_map=['r', 'y', 'k', 'g', 'b', 'm']):
    # label = [int(x) for x in label]
    if datasetName == 'uci' or datasetName == 'wisdm':
        color_map = ['r', 'y', 'k', 'g', 'b', 'm']
    elif datasetName == 'pamap2':
        color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FFC0CB', '#FFA500', '#800080', '#808080', '#00FF00']
    elif datasetName == 'unimib':
        color_map = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'lime', 'pink', 'teal', 'lavender', 'brown', 'beige', 'maroon', 'coral', 'olive']

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    # print(data, data.shape)
    # print(len(label))
    label1 = []
    for i in range(len(label)):
        print(len(label[i]))
        label1 =label1 +  [ int(x) for x in label[i]]

    # print(len(label))
    # print(data, data.shape)
    # print(len(label1))
    fig = plt.figure()
    for i in range(1100):

        #plt.plot(data[i, 0], data[i, 1], marker='o', markersize=1, color=color_map[label[i]])
        plt.plot(data[i+64, 0], data[i+64, 1], marker='o', markersize=1, color=color_map[label1[i]])
    plt.xticks([])
    plt.yticks([])
    # plt.title(title)
    plt.axis("off")
    # plt.legend()
    return fig


def TSNEplot(data, label, file_name, datasetName, type):
    # print("begin cal")

    # data = data.cpu().detach().numpy()
    if (label != None):
        # label = np.array(label.cpu())
        tsne_2D = TSNE(n_components=2, init='pca', random_state=1, perplexity=10)
        result_2D = tsne_2D.fit_transform(data)
        fig1 = plot_embedding_2D(result_2D, label, file_name, datasetName=datasetName)
        file_path = os.path.join(os.getcwd(), f"T-SNE_display\\{datasetName}\\")
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.savefig(os.path.join(file_path, f'{file_name}-{current_time}-{type}.pdf'), dpi=300)
        print("save ok!")
        fig1.show()
def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()

    Acc = []

    for (samples, targets) in metric_logger.log_every(data_loader, print_freq, header):


        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)


            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with amp_autocast():
            outputs = model(samples)

            if not args.cosub:
                # print(samples.shape)
                # print(outputs)
                targets = targets.squeeze()

                loss = criterion(outputs, targets.long())

        accuracy = accuracy_score(targets.detach().cpu(), np.argmax(outputs.detach().cpu(), axis=1))
        Acc.append(accuracy)
        if args.if_nan2num:
            with amp_autocast():
                loss = torch.nan_to_num(loss)

        loss_value = loss.item()


        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if isinstance(loss_scaler, timm.utils.NativeScaler):
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            if max_norm != None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    acc_avg = np.average(Acc)

    print('Accuary:'+ str(acc_avg))
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def confusion_matric(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

    ax.set_title("Confusion Matrix")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    matric_pred = []
    matric_true = []
    F1_list = []
    # switch to evaluation mode
    model.eval()

    Acc = []

    # for batch_idx, (images, target) in enumerate(data_loader):
    for sensor_data, target in metric_logger.log_every(data_loader, 10, header):
        sensors = sensor_data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with amp_autocast():
            output = model(sensors)


            output = output.to(device)
            target = target.to(device)
            # print(output.shape)
            # print(target.shape)
            target = target.squeeze(dim=1)

            loss = criterion(output.softmax(dim=1), target.long())
            # loss = criterion(output, target.long())
        f1 = f1_score(target.cpu(),torch.argmax(output.softmax(dim=1), dim=1).cpu(), average='weighted')
        F1_list.append(f1)
        acc1,acc2 = accuracy(output, target, topk=(1,2))
        accuracy1 = accuracy_score(target.detach().cpu(), np.argmax(output.detach().cpu(), axis=1))
        Acc.append(accuracy1)
        matric_pred = matric_pred + list(torch.argmax(output.softmax(dim=1), dim=1).cpu().numpy())
        matric_true = matric_true + list(target.cpu().numpy())
        batch_size = sensors.shape[0]

        metric_logger.update(loss=loss.item())

        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    F1_socre = np.average(F1_list)

    metric_logger.synchronize_between_processes()

    acc_score = np.average(Acc)
    print('* Acc {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    print('\n F1_score:'+str(F1_socre))
    print('\n Acc_score' + str(acc_score))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
