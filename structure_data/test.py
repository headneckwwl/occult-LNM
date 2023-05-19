
import argparse

import torch

from _algo.dnn.dnn_model import CostumeNet
from _algo.dnn.structured_ds import StructuredDataset
from _algo.dnn.train import split_dataloader, evaluate
from _algo.utils.about_log import logger


def batch_predict(args):
    structured_ds = StructuredDataset('bc_data.csv', 'bc_data.csv', fillna=0, label='diagnosis', keys='id',
                                      exclude=['diagnosis'], mode='joint',
                                      label_mapping_func=lambda x: 1 if x == 'M' else 0,
                                      as_onehot=False, n_classes=2)
    train_loader, validation_loader = split_dataloader(structured_ds, args.batch_size)
    config = {'input_dim': 60, 'hidden_unit': [64, 256, 256, 128], 'dropout': 0.2}
    model = CostumeNet(config, output_unit=2)
    model.load_state_dict(torch.load('BEST-training-params.pth', map_location='cpu')['model_state_dict'])
    eval_acc = evaluate(model, validation_loader, 'cpu')
    logger.info(f"__FINAL__\tBest Acc: {eval_acc}")


def single_predict(args):
    structured_ds = StructuredDataset('bc_data.csv', 'bc_data.csv', fillna=0, label='diagnosis', keys='id',
                                      exclude=['diagnosis'], mode='joint',
                                      label_mapping_func=lambda x: 1 if x == 'M' else 0,
                                      as_onehot=False, n_classes=2)
    train_loader, validation_loader = split_dataloader(structured_ds, args.batch_size)
    config = {'input_dim': 60, 'hidden_unit': [64, 256, 256, 128], 'dropout': 0.2}
    model = CostumeNet(config, output_unit=2)
    model.load_state_dict(torch.load('BEST-training-params.pth', map_location='cpu')['model_state_dict'])
    X, y = validation_loader.dataset[0]
    X = [torch.tensor([d], dtype=torch.float32) for d in X]
    print(torch.nn.functional.softmax(model(X), dim=1))
    print(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=' Structure Model')

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--init_lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', default='sgd', help='Optimizer')
    parser.add_argument('--retrain', default=None, help='Retrain from path')
    parser.add_argument('--model_root', default='.', help='path where to save')
    parser.add_argument('--iters_verbose', default=10, type=int, help='print frequency')
    parser.add_argument('--iters_start', default=0, type=int, help='Iters start')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained core or not')

    single_predict(parser.parse_args())
