
import argparse
import glob
import json
import os.path as osp
from typing import List
from PIL import Image
import numpy as np

import imgviz
import labelme
from termcolor import colored

from _algo.utils import truncate_dir, split_dataset, create_dir_if_not_exists
from _algo.utils.about_log import logger, log_long_str

__all__ = ['clf_covert2rec']


def clf_covert2rec(input_dir: str, save_dir: str, partition: List[float] = [0.8, 0.2], cmd=False):
    """
    Convert labelme format annotation to _algo training dataset.

    Args:
        input_dir: str, input dir
        save_dir: str, output dir
        partition: float list, train, valid, test partition.
        cmd : 是否打印command line信息。

    Returns:

    """
    truncate_dir(save_dir)
    logger.info(f"Creating dataset: {save_dir}")
    label_files = glob.glob(osp.join(input_dir, "*.json"))
    logger.info(f'一共获取到{len(label_files)}标注样本.')
    if len(label_files) < 500:
        logger.warning(f'我们发现数据量有点少（{len(label_files)}<500），可能会影响您模型训练的精度，但是不影响训练流程。')
    # Get total classes name.
    classes_name = sorted(json.loads(open(label_files[0], encoding='utf8').read())['flags'].keys())
    with open(osp.join(save_dir, 'labels.txt'), 'w', encoding='utf8') as labels_file:
        print('\n'.join(classes_name), file=labels_file)
    log_long_str(f"一共获取到{len(classes_name)}个标签.\n他们是 {', '.join(classes_name)}")

    img_root = create_dir_if_not_exists(osp.join(save_dir, 'images'))
    name_dict = {}
    true_label_set = set()
    train, val, _ = split_dataset(label_files, partition=partition)
    for subset, subset_name in zip([train, val], ['train', 'val']):
        logger.info(f"Creating {subset_name} dataset: {save_dir}")
        with open(osp.join(save_dir, subset_name + '.txt'), 'w', encoding='utf8') as record_file:
            for filename in subset:
                logger.info(f"\tGenerating dataset from: {filename}")
                label_file = labelme.LabelFile(filename=filename)
                # Save label and records
                for label_name, label_value in label_file.flags.items():
                    if label_value:
                        true_label_set.add(label_name)
                        break
                if sum(label_file.flags.values()) > 1:
                    logger.warning(rf'在{filename}发现了样本多余一个标签，我们将使用第一个标签({label_name})！')
                elif sum(label_file.flags.values()) == 0:
                    logger.warning(rf'在{filename}没有发现任何一个标签，我们将丢弃掉这个样本！')
                img = labelme.utils.img_data_to_arr(label_file.imageData)
                # Get output filename
                image_name = osp.basename(label_file.imagePath)
                if image_name not in name_dict:
                    name_dict[image_name] = 1
                else:
                    name_dict[image_name] += 1
                    image_name = image_name[:-4] + f'_{name_dict[image_name]}' + '.jpg'
                imgviz.io.imsave(osp.join(img_root, image_name), np.array(Image.fromarray(img).convert('RGB')))
                print(f"{image_name}\t{label_name}", file=record_file)
        no_samples = set(classes_name) - true_label_set
        if no_samples:
            raise ValueError(f"标签{', '.join(no_samples)}没有任何样本，这样将无法训练!")
    if cmd:
        train_file = osp.join(save_dir, 'train.txt')
        valid_file = osp.join(save_dir, 'val.txt')
        labels_file = osp.join(save_dir, 'labels.txt')
        data_pattern = osp.join(save_dir, 'images')

        hello_str = f"{'*' * 49} 使用下面命令，运行模型。 {'*' * 49}"
        end_str = f"{'*' * 42} 如遇问题欢迎联系微信：SingularityCloud {'*' * 42}"
        commands = [r'cd C:\Users\\.conda\envs\\Lib\site-packages\_algo/classification',
                    f'python run_classification.py --train {train_file} --valid {valid_file} --data_pattern {data_pattern} '
                    f'--labels_file {labels_file}']
        commands_str = colored('; '.join(commands), 'green', attrs=['bold'])
        print(f"{hello_str}\n\n{commands_str}\n\n{end_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert labelme labeled data to classification ListDataset')
    parser.add_argument("--input_dir", help="input annotated directory")
    parser.add_argument("--save_dir", help="output dataset directory")
    parser.add_argument("--partition", nargs='*', type=float, default=[0.8, 0.2], help="partition split into train val")
    args = parser.parse_args()

    clf_covert2rec(args.input_dir, args.save_dir, args.partition)
