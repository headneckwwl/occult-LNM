
import argparse

import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler

from _core.models.fusion.dnn_model import CostumeNet
from _algo.structure_dnn.structured_ds import StructuredDataset
from _algo.structure_dnn.custom_utils import metric_func_clf, metric_func_reg
from _core.core import create_optimizer, create_lr_scheduler, create_losses
from _algo.utils.about_log import logger


def evaluate(model: torch.nn.Module, data_loader, device, ):
    model.eval()
    y_all = []
    output_all = []
    metric_func = metric_func_clf if data_loader.dataset.n_classes is not None else metric_func_reg
    for X, y in data_loader:
        if isinstance(X, list):
            X = [d.to(device) for d in X]
        else:
            X = X.to(device)
        # y = torch.squeeze(torch.tensor(y, dtype=torch.int64).to(device))
        y = torch.squeeze(y).to(device)
        y_all.append(y)
        output = model(X)
        output_all.append(output)
    eval_acc = metric_func(torch.cat(output_all, dim=0), torch.cat(y_all, dim=0), onehot=False)
    logger.info(f'__EVAL__\tacc@1: {eval_acc}')
    return eval_acc


def train_one_epoch(model, criterion_fn, optimizer, data_loader, lr_scheduler, device, epoch):
    model.train()
    metric_func = metric_func_clf if data_loader.dataset.n_classes is not None else metric_func_reg
    with torch.set_grad_enabled(True):
        for iters, (X, y) in enumerate(data_loader):
            if isinstance(X, list):
                X = [d.to(device) for d in X]
            else:
                X = X.to(device)
            # y = torch.squeeze(torch.tensor(y, dtype=torch.int64).to(device))
            y = torch.squeeze(y).to(device)
            output = model(X)
            loss = criterion_fn(output, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            logger.info(f'__TRAIN__\tepoch: {epoch}, iters: {iters}, loss: {loss.item():.4f}, '
                        f'lr: {lr_scheduler.get_last_lr()[0]:.4f}, '
                        f'acc: {metric_func(output, y, onehot=False)}')


def split_dataloader(dataset, batch_size, validation_split=0.2, shuffle_dataset=True, random_seed=1234):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, drop_last=True)
    return train_loader, validation_loader


def main(args):
    structured_ds = StructuredDataset('bc_data.csv', 'bc_data.csv', fillna=0, label='diagnosis', keys='id',
                                      exclude=['diagnosis'], mode='joint',
                                      label_mapping_func=lambda x: 1 if x == 'M' else 0,
                                      as_onehot=False, n_classes=None)
    train_loader, validation_loader = split_dataloader(structured_ds, args.batch_size)
    config = {'input_dim': 60, 'hidden_unit': [64, 256, 256, 128], 'dropout': 0.2}
    model = CostumeNet(config, output_unit=1)
    optimizer = create_optimizer(args.optimizer, parameters=model.parameters(), lr=args.init_lr)
    lr_scheduler = create_lr_scheduler('cosine', optimizer,
                                       T_max=args.epochs * len(train_loader.dataset) // args.batch_size)
    # Setup the loss function
    criterion_fn = create_losses('mse')
    bst_acc = 0 if structured_ds.n_classes is not None else -99999
    for epoch in range(args.epochs):
        train_one_epoch(model, criterion_fn, optimizer, train_loader, lr_scheduler, 'cpu', epoch)
        eval_acc = evaluate(model, validation_loader, 'cpu')
        if structured_ds.n_classes is None:
            eval_acc = -eval_acc
        if eval_acc > bst_acc:
            bst_acc = eval_acc
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict()},
                       'BEST-training-params.pth')
    logger.info(f"__FINAL__\tBest Acc: {abs(bst_acc)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=' Structure Model')

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--init_lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', default='sgd', help='Optimizer')
    parser.add_argument('--retrain', default=None, help='Retrain from path')
    parser.add_argument('--model_root', default='.', help='path where to save')
    parser.add_argument('--iters_verbose', default=10, type=int, help='print frequency')
    parser.add_argument('--iters_start', default=0, type=int, help='Iters start')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained core or not')

    main(parser.parse_args())
