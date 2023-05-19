
import argparse
import glob
import os
import os.path as osp
import sys
from typing import List

import imgviz
import labelme

from _algo.utils import truncate_dir, split_dataset, create_dir_if_not_exists
from _algo.utils.about_log import logger, log_long_str

__all__ = ['obj_convert2voc']
try:
    import lxml.builder
    import lxml.etree
except ImportError:
    print("Please install lxml:\n\n    pip install lxml\n")
    sys.exit(1)


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


def obj_convert2voc(input_dir: str, save_dir: str, labels: List[str] = None, partition: List[float] = [0.8, 0.2],
                    noviz: bool = False, recursive: bool = True):
    """
    Convert labelme format annotation to _algo training dataset.

    Args:
        input_dir: str, input dir
        save_dir: str, output dir
        labels: str, labels file which contains labels of this dataset.
        partition: float list, train, valid, test partition.
        noviz: bool, viz or not.

    Returns:

    """
    truncate_dir(save_dir)
    logger.info(f"Creating dataset: {save_dir}")

    class_names = []
    class_name_to_id = {}

    label_files = get_labels_file(input_dir, recursive=recursive)
    # Get total classes name.
    if labels is not None:
        labels_list = labels
    else:
        labels_list = set()
        for filename in label_files:
            label_file = labelme.LabelFile(filename=filename)
            labels_list |= set([s['label'] for s in label_file.shapes])
        labels_list = sorted(list(labels_list))
        log_long_str(f"一共获取到{len(labels_list)}个标签.\n他们是 {', '.join(labels_list)}")

    if labels_list[0] != '__ignore__':
        labels_list.insert(0, '__ignore__')
    if labels_list[1] != '_background_':
        labels_list.insert(1, '_background_')
    for i, line in enumerate(labels_list):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    logger.info(f"class_names: {class_names}")
    out_class_names_file = osp.join(save_dir, "obj_labels.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    logger.info(f"Saved class_names: {out_class_names_file}")

    label_files = glob.glob(osp.join(input_dir, "*.json"))
    train, val, _ = split_dataset(label_files, partition=partition)
    for subset, subset_name in zip([train, val], ['train', 'val']):
        create_dir_if_not_exists(osp.join(save_dir, subset_name, "JPEGImages"))
        create_dir_if_not_exists(osp.join(save_dir, subset_name, "Annotations"))
        if not noviz:
            create_dir_if_not_exists(osp.join(save_dir, subset_name, "AnnotationsVisualization"))
        logger.info(f"Creating {subset_name} dataset: {save_dir}")
        for filename in subset:
            logger.info(f"Generating dataset from: {filename}")

            label_file = labelme.LabelFile(filename=filename)

            base = osp.splitext(osp.basename(filename))[0]
            out_img_file = osp.join(save_dir, subset_name, "JPEGImages", base + ".jpg")
            out_xml_file = osp.join(save_dir, subset_name, "Annotations", base + ".xml")
            if not noviz:
                out_viz_file = osp.join(save_dir, subset_name, "AnnotationsVisualization", base + ".jpg")

            img = labelme.utils.img_data_to_arr(label_file.imageData)
            imgviz.io.imsave(out_img_file, img)

            maker = lxml.builder.ElementMaker()
            xml = maker.annotation(
                maker.folder(),
                maker.filename(base + ".jpg"),
                maker.database(),  # e.g., The VOC2007 Database
                maker.annotation(),  # e.g., Pascal VOC2007
                maker.image(),  # e.g., flickr
                maker.size(
                    maker.height(str(img.shape[0])),
                    maker.width(str(img.shape[1])),
                    maker.depth(str(img.shape[2])),
                ),
                maker.segmented(),
            )

            bboxes = []
            labels = []
            for shape in label_file.shapes:
                if shape["shape_type"] != "rectangle":
                    print(
                        "Skipping shape: label={label}, "
                        "shape_type={shape_type}".format(**shape)
                    )
                    continue

                class_name = shape["label"]
                class_id = class_names.index(class_name)

                (xmin, ymin), (xmax, ymax) = shape["points"]
                # swap if min is larger than max.
                xmin, xmax = sorted([xmin, xmax])
                ymin, ymax = sorted([ymin, ymax])

                bboxes.append([ymin, xmin, ymax, xmax])
                labels.append(class_id)

                xml.append(
                    maker.object(
                        maker.name(shape["label"]),
                        maker.pose(),
                        maker.truncated(),
                        maker.difficult(),
                        maker.bndbox(
                            maker.xmin(str(xmin)),
                            maker.ymin(str(ymin)),
                            maker.xmax(str(xmax)),
                            maker.ymax(str(ymax)),
                        ),
                    )
                )

            if not noviz:
                captions = [class_names[label] for label in labels]
                viz = imgviz.instances2rgb(
                    image=img,
                    labels=labels,
                    bboxes=bboxes,
                    captions=captions,
                    font_size=15,
                )
                imgviz.io.imsave(out_viz_file, viz)

            with open(out_xml_file, "wb") as f:
                f.write(lxml.etree.tostring(xml, pretty_print=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", help="input annotated directory")
    parser.add_argument("--save_dir", help="output dataset directory")
    parser.add_argument('--labels', default='None', nargs='*', help='Labels')
    parser.add_argument("--partition", nargs='*', type=float, default=[0.8, 0.2], help="partition split into train val")
    parser.add_argument("--noviz", help="no visualization", action="store_true")
    args = parser.parse_args()
    obj_convert2voc(args.input_dir, args.save_dir, args.labels, args.partition, args.noviz)
