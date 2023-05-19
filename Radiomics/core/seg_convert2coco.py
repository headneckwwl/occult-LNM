
import argparse
import collections
import datetime
import glob
import json
import os
import sys
import time
import uuid
from typing import List

import imgviz
import labelme
import numpy as np
from PIL import Image
from termcolor import colored

from _algo.utils import split_dataset
from _algo.utils import truncate_dir, create_dir_if_not_exists
from _algo.utils.about_log import logger, log_long_str

__all__ = ['seg_covert2voc']

from _algo.utils.segmentation_utils import get_color_map_list

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_labels_file(input_dir, recursive: bool = True):
    if recursive:
        label_files = []
        for root, _, files in os.walk(input_dir):
            label_files.extend([os.path.join(root, f) for f in files if f.endswith('.json')])
    else:
        label_files = glob.glob(os.path.join(input_dir, "*.json"))
    logger.info(f'{input_dir}一共获取到{len(label_files)}标注样本.')
    if len(label_files) < 500:
        logger.warning(f'我们发现数据量有点少（{len(label_files)}<500），可能会影响您模型训练的精度，但是不影响训练流程。')
    return label_files


def seg_covert2voc(input_dir: str = None, save_dir: str = '.', partition: List[float] = [0.8, 0.2], noviz: bool = False,
                   labels: List[str] = None, train_dir=None, val_dir=None, recursive: bool = True,
                   del_directly: bool = False):
    """
    Convert labelme format annotation to _algo training dataset.

    Args:
        input_dir: str, input dir
        save_dir: str, output dir
        partition: float list, train, valid, test partition.
        noviz: bool, viz or not.
        labels: overlap list of str for each label or None,
        例如：labels = [label1, label2, label3]，在转化数据的时候，如果label之间发生重叠，则label3会覆盖label2,覆盖label1。
        train_dir:手动指定训练集
        val_dir: 手动指定验证集，可以是一个list，里面所有的数据都用来做测试
        recursive: 是否递归查找。默认为True
        del_directly: 是否直接删除。

    Returns:

    """
    assert input_dir is not None or train_dir is not None, 'input_dir和train_dir不能同时为空！'
    truncate_dir(save_dir, del_directly=del_directly)
    logger.info(f"Creating dataset: {save_dir}")

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None, )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )
    label_files = get_labels_file(input_dir or train_dir, recursive=recursive)
    # Get total classes name.
    if labels is not None:
        labels_list = labels
    else:
        labels_list = set()
        for filename in label_files:
            try:
                label_file = labelme.LabelFile(filename=filename)
            except Exception as e:
                logger.error(f'标签解析阶段，由于{e}，{filename}解析失败！')
                continue
            labels_list |= set([s['label'] for s in label_file.shapes])
        labels_list = sorted(list(labels_list))
        log_long_str(f"一共获取到{len(labels_list)}个标签.\n他们是 {', '.join(labels_list)}")

    # Convert label names to id
    class_name_to_id = {}
    if labels_list[0] != '__ignore__':
        labels_list.insert(0, '__ignore__')
    if labels_list[1] != '_background_':
        labels_list.insert(1, '_background_')
    for i, line in enumerate(labels_list):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        class_name_to_id[class_name] = class_id
        data["categories"].append(dict(supercategory=None, id=class_id, name=class_name, ))
    if input_dir:
        train, val, _ = split_dataset(label_files, partition=partition)
    else:
        train = label_files
        val = []
        for vd in val_dir:
            val.extend(get_labels_file(vd, recursive=recursive))
    for subset, subset_name in zip([train, val], ['train', 'val']):
        data['images'] = []
        data['annotations'] = []
        create_dir_if_not_exists(os.path.join(save_dir, subset_name, "JPEGImages"))
        create_dir_if_not_exists(os.path.join(save_dir, subset_name, "Masks"))
        if not noviz:
            create_dir_if_not_exists(os.path.join(save_dir, subset_name, "Visualization"))
        create_dir_if_not_exists(os.path.join(save_dir, subset_name))
        out_ann_file = os.path.join(os.path.join(save_dir, subset_name), "annotations.json")
        for image_id, filename in enumerate(subset):
            logger.info(f"Generating {subset_name} from: {filename}")
            try:
                label_file = labelme.LabelFile(filename=filename)
            except:
                logger.error(f'\t{filename}解析失败！')
                continue
            base = os.path.splitext(os.path.basename(filename))[0]
            out_img_file = os.path.join(save_dir, subset_name, "JPEGImages", base + ".jpg")
            out_msk_file = os.path.join(save_dir, subset_name, "Masks", base + ".png")
            if os.path.exists(out_img_file):
                out_img_file = os.path.join(save_dir, subset_name, "JPEGImages", f"{base}_{int(time.time())}" + ".jpg")
                out_msk_file = os.path.join(save_dir, subset_name, "Masks", f"{base}_{int(time.time())}" + ".png")
            img = labelme.utils.img_data_to_arr(label_file.imageData)
            imgviz.io.imsave(out_img_file, img)
            data["images"].append(
                dict(
                    license=0,
                    url=None,
                    file_name=os.path.relpath(out_img_file, os.path.dirname(out_ann_file)),
                    height=img.shape[0],
                    width=img.shape[1],
                    date_captured=None,
                    id=image_id,
                )
            )
            mask_arr = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            masks = {}  # for area
            segmentations = collections.defaultdict(list)  # for segmentation
            for shape in label_file.shapes:
                points = shape["points"]
                label = shape["label"]
                if label not in labels_list:
                    logger.warning(f'\t{label} 没有出现在{labels_list}, 我们将其标注信息丢弃！')
                    continue
                group_id = shape.get("group_id")
                shape_type = shape.get("shape_type", "polygon")
                try:
                    mask = labelme.utils.shape_to_mask(img.shape[:2], points, shape_type)
                except Exception as e:
                    logger.warning(f'\t{filename}中存在{e}解析失败。')
                    continue
                if group_id is None:
                    group_id = uuid.uuid1()

                instance = (label, group_id)

                if instance in masks:
                    masks[instance] = masks[instance] | mask
                else:
                    masks[instance] = mask

                if shape_type == "rectangle":
                    (x1, y1), (x2, y2) = points
                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])
                    points = [x1, y1, x2, y1, x2, y2, x1, y2]
                else:
                    points = np.asarray(points).flatten().tolist()

                segmentations[instance].append(points)
            segmentations = dict(segmentations)

            for instance, mask in masks.items():
                cls_name, group_id = instance
                if cls_name not in class_name_to_id:
                    continue
                cls_id = class_name_to_id[cls_name]
                mask_arr[mask] = cls_id
                mask = np.asfortranarray(mask.astype(np.uint8))
                mask = pycocotools.mask.encode(mask)
                area = float(pycocotools.mask.area(mask))
                bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

                data["annotations"].append(
                    dict(
                        id=len(data["annotations"]),
                        image_id=image_id,
                        category_id=cls_id,
                        segmentation=segmentations[instance],
                        area=area,
                        bbox=bbox,
                        iscrowd=0,
                    )
                )
            # 保存Mask索引图像
            colors = np.array(get_color_map_list(len(class_name_to_id))).reshape(-1).tolist()
            mask_img = Image.fromarray(mask_arr).convert('P')
            mask_img.putpalette(colors)
            mask_img.save(out_msk_file)
            # imgviz.io.imsave(out_msk_file, mask_arr)
            try:
                labels, captions, masks = zip(
                    *[
                        (class_name_to_id[cnm], cnm, msk)
                        for (cnm, gid), msk in masks.items()
                        if cnm in class_name_to_id
                    ]
                )
                if not noviz:
                    viz = imgviz.instances2rgb(
                        image=img,
                        labels=labels,
                        masks=masks,
                        captions=captions,
                        font_size=15,
                        line_width=2,
                    )
                    out_viz_file = os.path.join(save_dir, subset_name, "Visualization", os.path.basename(out_img_file))
                    imgviz.io.imsave(out_viz_file, viz)
            except:
                logger.error(f'可视化{base}出错, 不过此错误可以忽略!')
        with open(out_ann_file, "w", encoding='utf8') as f:
            json.dump(data, f, ensure_ascii=False, indent=True)
    with open(os.path.join(save_dir, 'seg_labels.txt'), 'w') as f:
        print('\n'.join(labels_list[1:]), file=f)

    hello_str = f"{'*' * 49} 使用下面命令，运行模型。 {'*' * 49}"
    end_str = f"{'*' * 42} 如遇问题欢迎联系微信：SingularityCloud {'*' * 42}"
    commands = [r'cd C:\Users\\.conda\envs\\Lib\site-packages\_algo/segmentation',
                f'python run_segmentation.py --dataset demo_coco_fmt --data_path {save_dir}']
    commands_str = colored('; '.join(commands), 'green', attrs=['bold'])
    print(f"{hello_str}\n\n{commands_str}\n\n{end_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", help="input annotated directory")
    parser.add_argument("--save_dir", help="output dataset directory")
    parser.add_argument("--partition", nargs='*', type=float, default=[0.8, 0.2], help="partition split into train val")
    parser.add_argument("--noviz", help="no visualization", action="store_true")
    parser.add_argument('--labels', default='None', nargs='*', help='Labels')
    args = parser.parse_args()
    seg_covert2voc(args.input_dir, args.save_dir, args.partition, args.noviz)
