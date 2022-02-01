#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script for learning a model.

A trained model must nerve on a sound photograph.
The script receives a dataset at the input for training,
consisting of images and their masks, where the location of the nerve is shown.
"""
import os

import segmentation_models_pytorch as smp
import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import configargparse
import logging
import pandas as pd

from dataset import SegmentDataset, DataLoader
from common.augmentation import (get_training_augmentation, get_preprocessing, get_validation_augmentation)
from utils.fs import clear_directory


def parse_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__, default_config_files=['./train_config.cfg'])

    parser.add_argument('--config', is_config_file=True, help='Path to config file')
    parser.add_argument('--lr', help='Learning rate', default=0.001, type=float)
    parser.add_argument('--epochs', help='Number of network learning speakers', default=10, type=int)
    parser.add_argument('--batch_size', help='How many parts to divide a set of images for training',
                        default=1, type=int)
    parser.add_argument('--num_workers', help='Num of threads for DataLoaders', default=1, type=int)
    parser.add_argument('--save_to', help='Path to dir saving models')
    parser.add_argument('--activation', help='function activation', default='sigmoid',
                        choices=['identity', 'sigmoid', 'softmax2d', 'softmax', 'logsoftmax'])
    parser.add_argument('--device', help='Device for load dataset', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--dir_train', required=True, help='path to save images for train net')
    parser.add_argument('--dir_valid', required=True, help='path to save images for validation net')
    parser.add_argument('--continues', help='continues network (path to model file or true str)', required=True)
    parser.add_argument('--metrics', help='network quality metrics', default='iou 0.5 rec 0.5 prec 0.5')
    parser.add_argument('--loss', help='loss type', choices=['jaccard, dice'], default='jaccard')

    return parser.parse_args()


def get_metrics(str_metrics):
    """
    Function for get network quality metrics
    Args:
        str_metrics: Str format: Metric1 threshold1 Metrics2 threshold2 ...

    Returns:
        Array metrics (Func)
    """
    metrics_data = str_metrics.split(' ')

    metrics_keys = metrics_data[::2]
    metrics_values = metrics_data[1::2]

    metrics = []

    for i in range(0, len(metrics_keys)):
        if metrics_keys[i] == 'iou':
            metrics.append(smp.utils.metrics.IoU(threshold=float(metrics_values[i])))
        elif metrics_keys[i] == 'rec':
            metrics.append(smp.utils.metrics.Recall(threshold=float(metrics_values[i])))
        elif metrics_keys[i] == 'prec':
            metrics.append(smp.utils.metrics.Precision(threshold=float(metrics_values[i])))

    return metrics


def get_continues_model(path):
    """
    Get model for continues train
    Args:
        path: path to dir models (search last model) or path to model file

    Returns:
        Dict
    """
    if os.path.isfile(path):
        data = torch.load(path)
    else:
        max_epoch = 0
        data = {}
        for root, _, files in os.walk(path):
            for filename in files:
                if not os.path.splitext(filename)[1] == '.pth':
                    continue
                load_data = torch.load(os.path.join(root, filename))

                if load_data['epoch'] > max_epoch:
                    max_epoch = load_data['epoch']
                    data = load_data

    return data


def save(
        epoch,
        metric,
        path,
        args,
        model,
        optimizer,
        device,
        model_name
):
    """
    Function for save best models and update statistic (best metrics)
    Args:
        epoch: Number of network learning speakers
        metric: Metrics for to assess the quality of network
        path: Path to save model
        args: Arguments command line
        model: Model for saving
        optimizer: Optimizer
        device: Device where loading model and images
        model_name: train or tests
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args,
        'encoder_name': 'resnet34',
        'encoder_weights': 'imagenet',
        'device': device,
        'classes': 1,
        'activation': args.activation,
        'metric': metric
    }, os.path.join(path, f'model_{model_name}_{metric["name"]}.pth'))

    metrics_csv = pd.read_csv(os.path.join(path, 'metrics.csv'), index_col='metrics')
    metrics_csv.loc[metric['name'], f'best_{model_name}_metrics'] = metric['value']
    metrics_csv.to_csv(os.path.join(path, 'metrics.csv'))

    logging.info(f'Model {model_name} saved ({metric["name"]})!')


def train(
        epochs,
        device,
        model,
        metrics,
        optimizer,
        loss,
        train_loader,
        validate_loader,
        args,
        prev_state,
        writer,
        save_to,
):
    """
    Function for train model and save the best versions models
    Args:
        epochs: Number of network learning speakers
        device: Device where loading model and images (cuda or cpu)
        model: Model for training
        metrics: Metrics for to assess the quality of network
        optimizer: Optimizer
        loss: Function Loss
        train_loader: Loader images for training network
        validate_loader: Loader images for testing work network
        args: Arguments command line
        prev_state: Previous state train (best statistic for metrics and training epochs)
        writer: Writer metrics (for torchvision)
        save_to: Path to save model
    """
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    best_train_metrics = prev_state['best_train_metrics']
    best_test_metrics = prev_state['best_test_metrics']

    loss_name = 'jaccard_loss' if args.loss == 'jaccard' else 'dice_loss'

    # training cycle
    for i in range(prev_state['epochs'], prev_state['epochs'] + epochs):
        print(f'\nEpoch: {i + 1} / {prev_state["epochs"] + epochs + 1}')

        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(validate_loader)

        # store metrics
        writer.add_scalars('IoU', {'train': train_logs['iou_score'], 'test': valid_logs['iou_score']}, i)
        writer.add_scalars('Recall', {'train': train_logs['recall'], 'test': valid_logs['recall']}, i)
        writer.add_scalars('Precision', {'train': train_logs['precision'], 'test': valid_logs['precision']}, i)
        writer.add_scalars('Loss', {'train': train_logs[loss_name], 'test': valid_logs[loss_name]}, i)
        writer.flush()

        for k, v in best_train_metrics.items():
            if k != 'dice_loss' and (k == loss_name and train_logs[k] < v or k != loss_name and train_logs[k] > v):
                best_train_metrics[k] = train_logs[k]

                save(
                    epoch=i,
                    path=save_to,
                    metric={'name': k, 'value': train_logs[k]},
                    args=args,
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    model_name='train'
                )

        for k, v in best_test_metrics.items():
            if k != 'dice_loss' and (k == loss_name and valid_logs[k] < v or k != loss_name and valid_logs[k] > v):
                best_test_metrics[k] = valid_logs[k]

                save(
                    epoch=i,
                    path=save_to,
                    metric={'name': k, 'value': valid_logs[k]},
                    args=args,
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    model_name='test'
                )


def main():
    """Application entry point."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    args = parse_args()

    # metrics
    metrics = get_metrics(args.metrics)

    dir_saving = f'models_{args.metrics.replace(" ", "_")}'
    save_to = os.path.join(args.save_to, dir_saving)

    os.makedirs(save_to, exist_ok=True)
    os.makedirs(os.path.join(save_to, 'logs'), exist_ok=True)
    clear_directory(os.path.join(save_to, 'logs'))

    writer = SummaryWriter(os.path.join(args.save_to, dir_saving, 'logs'))

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    prev_state = {
        'epochs': 0,
        'best_train_metrics': {'iou_score': 0.0, 'recall': 0.0, 'precision': 0.0,
                               'jaccard_loss': np.inf, 'dice_loss': np.inf},
        'best_test_metrics': {'iou_score': 0.0, 'recall': 0.0, 'precision': 0.0,
                              'jaccard_loss': np.inf, 'dice_loss': np.inf},
    }

    if not os.path.exists(os.path.join(save_to, 'metrics.csv')) or not args.continues:
        data = {
            'metrics': ['iou_score', 'recall', 'precision', 'jaccard_loss', 'dice_loss'],
            'best_train_metrics': [0.0, 0.0, 0.0, np.inf, np.inf],
            'best_test_metrics': [0.0, 0.0, 0.0, np.inf, np.inf],
        }

        df = pd.DataFrame(data)
        df.set_index('metrics', inplace=True)

        df.to_csv(os.path.join(save_to, 'metrics.csv'))

    # Get loss
    if args.loss == 'jaccard':
        loss = smp.utils.losses.JaccardLoss()
    else:
        loss = smp.utils.losses.DiceLoss()

    # Get model and optimizer
    if args.continues and args.continues != 'false' and len(os.listdir(save_to)) > 0:
        if args.continues == 'true':
            model_data = get_continues_model(save_to)
        else:
            model_data = get_continues_model(args.continues)
        logging.info('Model load!')

        prev_state['epochs'] = model_data['epoch']

        metrics_csv = pd.read_csv(os.path.join(save_to, 'metrics.csv'), index_col='metrics')

        for model_name, model_items in metrics_csv.items():
            for k, v in model_items.items():
                prev_state[model_name][k] = v

        model = smp.Unet(
            encoder_name=model_data['encoder_name'],
            encoder_weights=model_data['encoder_weights'],
            classes=model_data['classes'],
            activation=model_data['activation'],
        )

        model.load_state_dict(model_data['model_state_dict'])

        if model_data['device'] == torch.device('cuda') and torch.cuda.is_available():
            model.cuda()

        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), lr=args.lr),
        ])

        optimizer.load_state_dict(model_data['optimizer_state_dict'])

        preprocessing_fn = smp.encoders.get_preprocessing_fn(model_data['encoder_name'],
                                                             pretrained=model_data['encoder_weights'])
    else:
        model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            classes=1,
            activation=args.activation,
        )

        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), lr=args.lr),
        ])

        preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet34', pretrained='imagenet')

    # Get datasets
    train_dataset = SegmentDataset(
        args.dir_train + 'img/',
        args.dir_train + 'mask/',
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    valid_dataset = SegmentDataset(
        args.dir_valid + 'img/',
        args.dir_valid + 'mask/',
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    # Get loaders data
    train_loader = DataLoader(train_dataset,
                              batch_size=int(args.batch_size),
                              shuffle=True,
                              num_workers=int(args.num_workers))
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Show information training before start
    logging.info(f'''Starting training:
        Model: UNet
        Device: {device}
        Epochs: {args.epochs}
        Batch Size: {args.batch_size}
        Learning Rate: {args.lr}
        Continues Train: {True if args.continues else False}
        Training Size: {len(train_dataset)}
        Validation Size: {len(valid_dataset)}
        Function Activate: {args.activation}
        Save Path: {save_to or False}
    ''')

    train(
        epochs=args.epochs,
        device=device,
        model=model,
        metrics=metrics,
        optimizer=optimizer,
        loss=loss,
        train_loader=train_loader,
        validate_loader=valid_loader,
        args=args,
        prev_state=prev_state,
        writer=writer,
        save_to=save_to,
    )


if __name__ == '__main__':
    main()
