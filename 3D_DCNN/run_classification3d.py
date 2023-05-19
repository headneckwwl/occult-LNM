
import argparse
import os
import random
import shutil

from _algo.utils import create_dir_if_not_exists

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    AddChannel,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
    EnsureType,
)
from torch.utils.tensorboard import SummaryWriter

from _algo.datasets import create_classification_dataset
from _algo.utils.about_log import logger, log_long_str
from _core.core import create_lr_scheduler, create_optimizer, create_losses, create_model


def __config_list_or_folder_dataset(records, data_pattern):
    if not isinstance(records, (list, tuple)):
        records = [records]
    if records:
        if all(os.path.exists(r) for r in records) and all(os.path.isdir(r) for r in records):
            return {'records': None, 'ori_img_root': records}
        elif all(os.path.isfile(r) for r in records):
            return {'records': records, 'ori_img_root': data_pattern}
    else:
        return {'records': None, 'ori_img_root': data_pattern}
    raise ValueError(f"records({records}) or data_pattern({data_pattern}) config error!")


def train_one_epoch(model, device, train_loader: DataLoader, criterion, optimizer, lr_scheduler, writer, epoch,
                    **kwargs):
    model.train()
    epoch_loss = 0
    step = 0
    iters_verbose = kwargs.get('iters_verbose', 1)
    train_dir = kwargs.get('train_dir', './train')
    os.makedirs(train_dir, exist_ok=True)
    train_file = open(os.path.join(train_dir, f'Epoch-{epoch}.txt'), 'w')
    fnames = train_loader.dataset.image_files
    fname_idx = 0
    for idx, batch_data in enumerate(train_loader):
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(outputs.argmax(dim=1), labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        epoch_loss += loss.item()
        epoch_len = len(train_loader.dataset) // train_loader.batch_size
        if idx % iters_verbose == 0:
            logger.info(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        probabilities, predictions = torch.max(nn.functional.softmax(outputs, dim=1), 1)
        for idx_, (probability, prediction, label) in enumerate(zip(probabilities, predictions, labels)):
            train_file.write('%s\t%.3f\t%d\t%d\n' % (os.path.basename(fnames[fname_idx + idx_]),
                                                     probability, prediction, label))
        fname_idx += train_loader.batch_size
    epoch_loss /= step
    logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    return epoch_loss


def evaluate(model, device, val_loader, writer, epoch, **kwargs):
    model.eval()
    val_dir = kwargs.get('val_dir', './valid')
    os.makedirs(val_dir, exist_ok=True)
    val_file = open(os.path.join(val_dir, f'Epoch-{epoch}.txt'), 'w')
    fnames = val_loader.dataset.image_files
    fname_idx = 0
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        for val_data in val_loader:
            inputs, labels = val_data[0].to(device), val_data[1].to(device)
            val_outputs = model(inputs)
            value = torch.eq(val_outputs.argmax(dim=1), labels)
            # print(val_outputs.argmax(dim=1), val_labels)
            metric_count += len(value)
            num_correct += value.sum().item()
            probabilities, predictions = torch.max(nn.functional.softmax(val_outputs, dim=1), 1)
            for idx_, (probability, prediction, label) in enumerate(zip(probabilities, predictions, labels)):
                val_file.write('%s\t%.3f\t%d\t%d\n' % (os.path.basename(fnames[fname_idx + idx_]),
                                                       probability, prediction, label))
            fname_idx += val_loader.batch_size
        metric = num_correct / metric_count
        writer.add_scalar("val_accuracy", metric, epoch + 1)
    return metric


def main(args):
    train_datasets = create_classification_dataset(
        max2use=args.max2use, recursive=True, transform=None, labels_file=args.labels_file,
        **__config_list_or_folder_dataset(args.train, args.data_pattern))
    data = list(zip(train_datasets.file_names, train_datasets.labels))
    random.shuffle(data)
    train_images, train_labels = zip(*data)

    val_datasets = create_classification_dataset(
        max2use=args.max2use, recursive=True, transform=None, labels_file=args.labels_file,
        **__config_list_or_folder_dataset(args.val, args.data_pattern))
    data = list(zip(val_datasets.file_names, val_datasets.labels))
    random.shuffle(data)
    val_images, val_labels = zip(*data)
    log_long_str('Train:%s\nValid:%s' % (train_datasets, val_datasets))
    # Define transforms
    train_transforms = Compose([ScaleIntensity(), AddChannel(), Resize(args.roi_size), RandRotate90(), EnsureType()])
    val_transforms = Compose([ScaleIntensity(), AddChannel(), Resize(args.roi_size), EnsureType()])

    # create a training data loader
    train_ds = ImageDataset(image_files=train_images, labels=train_labels, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.j, pin_memory=torch.cuda.is_available())
    # create a validation data loader
    val_ds = ImageDataset(image_files=val_images, labels=val_labels, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available(), shuffle=False)

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(f'classification3d.{args.model_name}', n_input_channels=1,
                         num_classes=train_datasets.num_classes, pretrained=args.pretrained).to(device)
    # model = monai.networks.nets.DenseNet121().to(device)
    loss_function = create_losses('softmax_ce')
    optimizer = create_optimizer(args.optimizer, parameters=model.parameters(), lr=args.init_lr)
    lr_scheduler = create_lr_scheduler('cosine', optimizer,
                                       T_max=args.epochs * len(train_datasets) // args.batch_size)

    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    model_dir = create_dir_if_not_exists(os.path.join(args.model_root, args.model_name), add_date=True)
    create_dir_if_not_exists(os.path.join(model_dir, 'runs'))
    writer = SummaryWriter(log_dir=os.path.join(model_dir, 'runs'))
    max_epochs = args.epochs
    for epoch in range(max_epochs):
        logger.info(f"epoch {epoch + 1}/{max_epochs}")
        epoch_loss = train_one_epoch(model, device, train_loader, loss_function, optimizer, lr_scheduler,
                                     writer=writer, epoch=epoch, iters_verbose=args.iters_verbose,
                                     train_dir=os.path.join(model_dir, 'train'))
        epoch_loss_values.append(epoch_loss)
        if (epoch + 1) % args.val_interval == 0:
            metric = evaluate(model, device, val_loader, writer=writer, epoch=epoch,
                              val_dir=os.path.join(model_dir, 'valid'))
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                create_dir_if_not_exists(os.path.join(model_dir, 'viz'))
                torch.save({'state_dict': model.state_dict()},
                           os.path.join(model_dir, 'viz', f"BEST-training-params.pth"))
                shutil.copy(os.path.join(model_dir, 'train', f'Epoch-{epoch}.txt'),
                            os.path.join(model_dir, 'viz', 'BST_TRAIN_RESULTS.txt'))
                shutil.copy(os.path.join(model_dir, 'valid', f'Epoch-{epoch}.txt'),
                            os.path.join(model_dir, 'viz', 'BST_VAL_RESULTS.txt'))
            logger.info(f"current epoch: {epoch + 1} current accuracy: {metric:.4f} "
                        f"best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
    logger.info(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


DATA_ROOT = '/Users/zhangzhiwei/Downloads/Skull'
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--train', nargs='*', default=os.path.join(DATA_ROOT, 'train'), help='Training dataset')
    parser.add_argument('--val', nargs='*', default=os.path.join(DATA_ROOT, 'val'), help='Validation dataset')
    parser.add_argument('--labels_file', default=os.path.join(DATA_ROOT, 'labels.txt'), help='Labels file')
    parser.add_argument('--data_pattern', default=None, nargs='*', help='Where to save origin image data.')
    parser.add_argument('-j', '--worker', dest='j', default=0, type=int, help='Number of workers.(default=1)')
    parser.add_argument('--max2use', default=None, type=int, help='Maximum number of sample per class to be used!')
    parser.add_argument('--normalize_method', default='imagenet', choices=['-1+1', 'imagenet'],
                        help='Normalize method.')
    parser.add_argument('--model_name', default='resnet18', help='Model name')
    parser.add_argument('--roi_size', default=[96, 96, 96], type=int, nargs='*', help='Model name')
    parser.add_argument('--gpus', type=int, nargs='*', default=[0], help='GPU index to be used!')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('--init_lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', default='adam', help='Optimizer')
    parser.add_argument('--retrain', default=None, help='Retrain from path')
    parser.add_argument('--model_root', default='.', help='path where to save')
    parser.add_argument('--val_interval', default=1, type=int, help='val_interval')
    parser.add_argument('--iters_start', default=0, type=int, help='Iters start')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained core or not')

    main(parser.parse_args())
