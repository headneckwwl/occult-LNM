
import argparse
import copy
import json
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import shutil
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils as utils
import torch.utils.data
import torch.utils.data.distributed
from torch import Tensor

from _algo.datasets.ClassificationDataset import create_classification_dataset
from _core.core import create_model, create_optimizer, create_lr_scheduler
from _core.core import create_standard_image_transformer
from _core.core.losses_factory import create_losses
from _algo.utils import create_dir_if_not_exists, truncate_dir
from _algo.utils.about_log import ColorPrinter
from _algo.utils.about_log import logger, log_long_str

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train_model(model, device, dataloaders, batch_size, criterion, optimizer, lr_scheduler, num_epochs=25,
                iters_start=0, iters_verbose=10, save_dir='./', is_inception=False, save_per_epoch=False, **kwargs):
    """
    Train and valid core.

    :param model: The core generate from `create_model`.
    :param device: Which device to be used!
    :param dataloaders: Dataset loaders, `train` and `valid` can be specified.
    :param batch_size: Batch size.
    :param criterion: Loss function definition. Generated from `create_losses`
    :param optimizer: Optimizer. Generated from `create_optimizer`
    :param lr_scheduler: Learning scheduler. Generated from `create_lr_scheduler`
    :param num_epochs: Number of epochs
    :param iters_start: Iters start, default 0
    :param iters_verbose: Iters to display log. default 10.
    :param save_dir: Where to save core.
    :param is_inception: Is inception core, for different image resolution.
    :param save_per_epoch: 是否每个Epoch都保存。
    :return: best core parameters, validation accuracy.
    """
    time_since = time.time()
    time_verbose = time.time()
    valid_time_since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = None
    iters_done = 0
    cp = ColorPrinter()

    train_dir = truncate_dir(os.path.join(save_dir, 'train'), del_directly=True)
    valid_dir = truncate_dir(os.path.join(save_dir, 'valid'), del_directly=True)
    viz_dir = truncate_dir(os.path.join(save_dir, 'viz'), del_directly=True)
    train_log = [('Epoch', 'Iters', 'Loss', 'Acc@1')]
    train_file = None
    valid_file = None

    for epoch in range(num_epochs):
        epoch_iters_done = 0
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set core to training mode
                train_file = open(os.path.join(train_dir, f'Epoch-{epoch}.txt'), 'w')
            else:
                model.eval()  # Set core to evaluate mode
                valid_time_since = time.time()
                valid_file = open(os.path.join(valid_dir, f'Epoch-{epoch}.txt'), 'w')

            running_loss = 0.0
            running_corrects = 0
            running_size = 0
            # Iterate over data.
            for inputs, labels, fnames in dataloaders[phase]:
                inputs = inputs.to(device)
                input_size = inputs.shape[0]
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # Multi head outputs for inception.
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        if not isinstance(outputs, Tensor):
                            loss1 = criterion(outputs.logits, labels)
                            loss2 = criterion(outputs.aux_logits2, labels)
                            loss3 = criterion(outputs.aux_logits1, labels)
                            loss = loss1 + 0.4 * loss2 + 0.2 * loss3
                        else:
                            loss = criterion(outputs, labels)

                    if isinstance(outputs, Tensor):
                        probabilities, predictions = torch.max(nn.functional.softmax(outputs, dim=1), 1)
                    else:
                        probabilities, predictions = torch.max(nn.functional.softmax(outputs.logits, dim=1), 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        iters_done += 1
                        epoch_iters_done += 1
                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()
                        for fname, probability, prediction, label in zip(fnames, probabilities, predictions, labels):
                            train_file.write('%s\t%.3f\t%d\t%d\n' % (fname, probability, prediction, label))
                    else:
                        for fname, probability, prediction, label in zip(fnames, probabilities, predictions, labels):
                            valid_file.write('%s\t%.3f\t%d\t%d\n' % (fname, probability, prediction, label))
                running_loss += loss.item()
                running_corrects += int(torch.sum(predictions == labels.data))
                running_size += input_size
                if epoch_iters_done % iters_verbose == 0 and phase == 'train':
                    iters_epoch = (iters_done + iters_start) * batch_size / len(dataloaders[phase].dataset)
                    total_time = int(time.time() - time_since)
                    speed_info = iters_verbose * input_size / (time.time() - time_verbose)
                    hours_ = total_time // 3600
                    minutes_ = total_time // 60 % 60
                    acc_ = running_corrects * 100 / running_size
                    info = 'Phase: {phase}\tEpoch: {epoch}\tLR: {lr}\tLoss: {loss}\tAcc: {acc}\t ' \
                           'Speed: {speed} img/s\tTime: {time}'
                    logger.info(info.format(phase=cp.color_text(phase, color='green' if phase == 'train' else 'cyan'),
                                            epoch=cp.color_text('%.3f,%d' % (iters_epoch, iters_done), 'magenta'),
                                            lr=cp.color_text(','.join(['%.6f' % lr_ for
                                                                       lr_ in lr_scheduler.get_last_lr()]), 'cyan'),
                                            loss=cp.color_text('%.3f' % (running_loss / iters_verbose), 'red'),
                                            acc=cp.color_text('%.4f' % acc_, 'green' if acc_ > 70 else 'yellow'),
                                            speed=cp.color_text('%.2f' % speed_info, 'yellow'),
                                            time=cp.color_text(f'{hours_}hrs:{minutes_}min')))
                    train_log.append((iters_epoch, iters_done, running_loss / iters_verbose, acc_))
                    time_verbose = time.time()
                    running_loss = 0.0
                    running_corrects = 0
                    running_size = 0

            epoch_acc = running_corrects * 100 / len(dataloaders[phase].dataset)
            if phase == 'train':
                train_file.close()
                if save_per_epoch:
                    torch.save({'global_step': iters_done + iters_start,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lr_scheduler_state_dict': lr_scheduler.state_dict()},
                               os.path.join(train_dir, f'training-params-{iters_done + iters_start}.pth'))
            else:
                valid_file.close()
                val_acc_history.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy({'global_step': iters_done + iters_start,
                                                    'model_state_dict': model.state_dict(),
                                                    'optimizer_state_dict': optimizer.state_dict(),
                                                    'lr_scheduler_state_dict': lr_scheduler.state_dict()})
                    # Copy best model's validation
                    shutil.copyfile(os.path.join(valid_dir, f'Epoch-{epoch}.txt'),
                                    os.path.join(viz_dir, f'BST_VAL_RESULTS.txt'))
                    shutil.copyfile(os.path.join(train_dir, f'Epoch-{epoch}.txt'),
                                    os.path.join(viz_dir, f'BST_TRAIN_RESULTS.txt'))
                info = 'Phase: {phase}\tAcc: {acc}\tSpeed: {speed}img/s\tTime: {time}s'
                speed = len(dataloaders[phase].dataset) / (time.time() - valid_time_since)
                logger.info(info.format(phase=cp.color_text('valid', 'cyan'),
                                        acc=cp.color_text("%.3f" % epoch_acc),
                                        speed=cp.color_text("{:.4f}".format(speed), 'red'),
                                        time=cp.color_text('%.2f' % (time.time() - valid_time_since))))
    # Save training log to csv
    pd.DataFrame(train_log).to_csv(os.path.join(viz_dir, 'training_log.txt'),
                                   header=False, index=False, encoding='utf8')
    # Save labels file to viz directory
    shutil.copyfile(kwargs['labels_file'], os.path.join(viz_dir, 'labels.txt'))

    # Save the best training parameters.
    torch.save(best_model_wts, os.path.join(viz_dir, 'BEST-training-params.pth'))
    # load best core weights
    model.load_state_dict(best_model_wts['model_state_dict'])

    # Save task settings.
    with open(os.path.join(viz_dir, 'task.json'), 'w') as task_file:
        kwargs['task_spec'].update({"acc": best_acc, 'best_epoch': best_epoch})
        print(json.dumps(kwargs['task_spec'], indent=True, ensure_ascii=False), file=task_file)
    return model, val_acc_history


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


def main(args):
    assert args.gpus is None or args.batch_size % len(args.gpus) == 0, 'Batch size must exactly divide number of gpus'
    cp = ColorPrinter()
    if 'inception' in args.model_name.lower():
        image_size = (299, 299)
    else:
        image_size = (224, 224)
    # Initialize data transformer for this run
    kwargs = {}
    try:
        roi_size = args.__getattribute__('roi_size')
        kwargs['in_channels'] = roi_size[0]
    except:
        roi_size = None
    data_transforms = {'train': create_standard_image_transformer(image_size, phase='train',
                                                                  normalize_method=args.normalize_method,
                                                                  is_nii=True if roi_size is not None else False,
                                                                  roi_size=roi_size),
                       'valid': create_standard_image_transformer(image_size, phase='valid',
                                                                  normalize_method=args.normalize_method,
                                                                  is_nii=True if roi_size is not None else False,
                                                                  roi_size=roi_size)}
    # Initialize datasets and dataloader for this run
    image_datasets = {'train': create_classification_dataset(
        max2use=args.max2use, recursive=True, transform=data_transforms['train'], labels_file=args.labels_file,
        batch_balance=args.batch_balance, **__config_list_or_folder_dataset(args.train, args.data_pattern))}
    # Save labels file if needed!
    # save_classification_dataset_labels(image_datasets['train'], os.path.join(args.model_root, 'labels.txt'))
    image_datasets['valid'] = create_classification_dataset(
        max2use=args.val_max2use, transform=data_transforms['valid'], classes=image_datasets['train'].classes,
        recursive=True, **__config_list_or_folder_dataset(args.valid, args.data_pattern))
    assert image_datasets['train'].classes == image_datasets['valid'].classes
    log_long_str('Train:%s\nValid:%s' % (image_datasets['train'], image_datasets['valid']))
    # Initialize the core for this run
    logger.info('Creating model...')
    try:
        kwargs.update({'e2e_comp': args.e2e_comp})
    except:
        pass

    model = create_model(args.model_name, num_classes=image_datasets['train'].num_classes,
                         pretrained=args.pretrained if not args.retrain else False, **kwargs)
    # Send the core to GPU
    if args.gpus and len(args.gpus) > 1:
        model_ft = nn.DataParallel(model, device_ids=args.gpus)
    elif len(args.gpus) == 1:
        device = torch.device(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu")
        model_ft = model.to(device)
    else:
        model_ft = model.to('cpu')
    dataloaders_dict = {x: utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batch_size,
                                                 drop_last=False,
                                                 shuffle=True,
                                                 num_workers=args.j)
                        for x in ['train', 'valid']}
    # Initialize optimizer and learning rate for this run
    optimizer = create_optimizer(args.optimizer, parameters=model_ft.parameters(), lr=args.init_lr)
    lr_scheduler = create_lr_scheduler('cosine', optimizer,
                                       T_max=args.epochs * len(image_datasets['train']) // args.batch_size)
    # Setup the loss function
    criterion = create_losses('softmax_ce')

    if args.retrain and os.path.exists(args.retrain):
        state_dict = torch.load(args.retrain, map_location=device)
        model_ft.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        lr_scheduler.load_state_dict(state_dict['lr_scheduler_state_dict'])
        logger.info(cp.color_text(f"Successfully retrained from {args.retrain}"))
    # Train and evaluate
    model_dir = create_dir_if_not_exists(args.model_root, add_date=args.add_date)
    logger.info('Start training...')
    model_ft, hist = train_model(model_ft, device, dataloaders_dict, args.batch_size, criterion, optimizer,
                                 lr_scheduler, num_epochs=args.epochs,
                                 is_inception=("inception" in args.model_name),
                                 save_dir=os.path.join(model_dir, args.model_name),
                                 iters_verbose=args.iters_verbose,
                                 labels_file=args.labels_file,
                                 task_spec={'num_classes': image_datasets['train'].num_classes,
                                            'model_name': args.model_name,
                                            'transform': {'input_size': image_size,
                                                          'normalize_method': args.normalize_method},
                                            'device': str(device),
                                            'type': 'what'},
                                 save_per_epoch=args.save_per_epoch)

    logger.info(f'Done for training. Best is {max(hist)}@{hist.index(max(hist)) + 1} epoch. '
                f'Average accuracy for last 5 iters is {sum(hist[-5:]) / len(hist[-5:])}')


# Modify this parameter if necessary!
DATA_ROOT = os.path.expanduser(r'C:\Users\\Project\DS\CT')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--train', nargs='*', default=os.path.join(DATA_ROOT, 'train.txt'), help='Training dataset')
    parser.add_argument('--valid', nargs='*', default=os.path.join(DATA_ROOT, 'val.txt'), help='Validation dataset')
    parser.add_argument('--labels_file', default=os.path.join(DATA_ROOT, 'labels.txt'), help='Labels file')
    parser.add_argument('--data_pattern', default=os.path.join(DATA_ROOT, 'images'), nargs='*',
                        help='Where to save origin image data.')
    parser.add_argument('--roi_size', default=[32, 32, 32], nargs='*', type=int, help='Multi channels settings.')
    parser.add_argument('-j', '--worker', dest='j', default=4, type=int, help='Number of workers.(default=0)')
    parser.add_argument('--max2use', default=None, type=int, help='Maximum number of sample per class to be used!')
    parser.add_argument('--val_max2use', default=None, type=int, help='Maximum number of samples in valid phase'
                                                                      ' per class to be used!')
    parser.add_argument('--batch_balance', default=False, action='store_true', help='Batch balance samples for train.')
    parser.add_argument('--normalize_method', default='imagenet', choices=['-1+1', 'imagenet'],
                        help='Normalize method.')
    parser.add_argument('--model_name', default='resnet18', help='Model name')
    parser.add_argument('--gpus', type=int, nargs='*', default=[0], help='GPU index to be used!')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=3, type=int, help='number of total epochs to run')
    parser.add_argument('--init_lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', default='sgd', help='Optimizer')
    parser.add_argument('--retrain', default=None, help='Retrain from path')
    parser.add_argument('--model_root', default='models', help='path where to save')
    parser.add_argument('--add_date', default=False, action='store_true', help='是否在model_roo下添加日期。')
    parser.add_argument('--iters_verbose', default=1, type=int, help='print frequency')
    parser.add_argument('--iters_start', default=0, type=int, help='Iters start')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained core or not')
    parser.add_argument('--save_per_epoch', action='store_true', help='是否每个Epoch都保存，默认False')

    main(parser.parse_args())
