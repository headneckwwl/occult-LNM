
import abc
import collections
import json
import os
import os.path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from _algo.classification.eval_classification import init as init_clf, inference as inference_clf
from _algo.detection.eval_detection import init as init_obj, inference as inference_obj
from _algo.scripts.core import draw_roc, draw_confusion_matrix
from _algo.segmentation.eval_segmentation import init as init_seg, inference as inference_seg
from _algo.utils import create_dir_if_not_exists

matplotlib.pyplot.switch_backend('Agg')

__all__ = ['ClassificationVisualize', 'DetectionVisualize', 'SegmentationVisualize']
NECESSARY_FILES = ['BEST-training-params.pth', 'BST_VAL_RESULTS.txt', 'labels.txt', 'task.json', 'training_log.txt']
TaskTuple = collections.namedtuple('TaskTuple', ['sub_viz_dir', 'model_path', '_results',
                                                 'labels', 'label_path', 'task_params', 'training_log'])


def is_valid_config_path(config_path):
    if os.path.exists(config_path) and os.path.isdir(config_path):
        if all(nf in os.listdir(config_path) for nf in NECESSARY_FILES):
            return True
    return False


def get_log(model_root, task_type=None):
    if not is_valid_config_path(model_root):
        print(f'model_root={model_root} 参数必须是目录。', '可能存在的原因是任务没有训练完成，或者版本不对应。')
    assert task_type in ['what', 'where', 'which'], "目前仅支持中的what、where、which类任务分析！"
    model_path = os.path.join(model_root, 'BEST-training-params.pth')
    _results = os.path.join(model_root, 'BST_VAL_RESULTS.txt')
    labels = [l.strip() for l in open(os.path.join(model_root, 'labels.txt'), encoding='utf8')]
    label_path = os.path.join(model_root, 'labels.txt')
    training_log = os.path.join(model_root, 'training_log.txt')
    task_params = json.loads(open(os.path.join(model_root, 'task.json'), encoding='utf8').read())
    if task_type is None or task_params['type'] == task_type:
        return TaskTuple(os.path.abspath(model_root), model_path, _results, labels, label_path,
                         task_params, training_log)
    else:
        return None


class TaskVisualize(object):
    model: Dict[str, List]

    def __init__(self, model_root, single: bool = False, task_type=None, save_dir: str = './'):
        assert os.path.isdir(model_root), f'model_root={model_root} 参数必须是目录'
        self.model_root = model_root
        self.model = {}
        self.save_dir = save_dir
        self.task_type = task_type
        if single:
            self.sub_viz = {os.path.basename(model_root): get_log(model_root)}
            self.model[os.path.basename(model_root)] = None
        else:
            self.get_valid_config_path()

    def get_valid_config_path(self, model_root=None):
        self.sub_viz = {}
        model_root = model_root or self.model_root
        for root, sub_dir, _ in os.walk(model_root):
            # sub_model_root = os.path.join(root, sub_dir, 'viz')
            sub_model_root = root
            if is_valid_config_path(sub_model_root):
                sub_name = root[root.find(model_root) + len(model_root):]
                sub_name = sub_name.strip('/').strip(r'\\')
                sub_name = sub_name.replace('\\', '_').replace('/', '_')
                create_dir_if_not_exists(os.path.join(self.save_dir, sub_name))
                sub_viz_ = get_log(sub_model_root, task_type=self.task_type)
                if sub_viz_ is not None:
                    self.sub_viz[sub_name] = sub_viz_
                    self.model[sub_name] = None
                # if self.labels is None:
                #     self.labels = open(os.path.join(model_root, sub_dir, 'viz'))

    @abc.abstractmethod
    def viz(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, sample, model_name):
        raise NotImplementedError()


class ClassificationVisualize(TaskVisualize):
    TRAINING = ['Loss', 'Acc@1']

    def __init__(self, model_root, single: bool = False, save_dir: str = './'):
        super().__init__(model_root, single, task_type='what', save_dir=save_dir)
        self.results = {k_: pd.read_csv(v_.training_log, header=0, index_col=1)[ClassificationVisualize.TRAINING]
                        for k_, v_ in self.sub_viz.items()}

    def viz(self):
        viz_result = []
        # plot training log
        for k, v in self.results.items():
            viz_spec = {'model_name': f"what.{k}", "result_list": []}
            for name in ClassificationVisualize.TRAINING:
                
                v[name].plot(subplots=True, figsize=(16, 9))
                plt.legend(loc='lower right')
                plt.savefig(os.path.join(self.save_dir, k, f"{name}.svg"), dpi=300)
                plt.close()
                viz_spec['result_list'].append({'img': os.path.join(self.save_dir, k, f"{name}.svg"),
                                                'desc': f"{name}曲线"})
            if self.sub_viz[k].task_params['num_classes'] == 2:
                auc = draw_roc(self.sub_viz[k]._results, save_dir=os.path.join(self.save_dir, k))
                viz_spec['result_list'].append({'img': os.path.join(self.save_dir, k, f"roc.svg"),
                                                'desc': f"ROC曲线, auc={auc}"})
            draw_confusion_matrix(self.sub_viz[k]._results, self.sub_viz[k].label_path,
                                  save_dir=os.path.join(self.save_dir, k),
                                  num_classes=self.sub_viz[k].task_params['num_classes'])
            viz_spec['result_list'].append({'img': os.path.join(self.save_dir, k, f"confusion_matrix.svg"),
                                            'desc': f"混淆矩阵"})
            viz_result.append(viz_spec)
        return viz_result

    def predict(self, sample, model_name):
        assert model_name in self.model
        if self.model[model_name] is None:
            self.model[model_name] = init_clf(self.sub_viz[model_name].sub_viz_dir)
        results = inference_clf(sample, *self.model[model_name])
        return [{"img": os.path.join(self.save_dir, model_name, r[0]),
                 'desc': f"预测标签:{r[-1]},详细结果:"
                         f"{json.dumps({k_: f'{v_:.2f}' for k_, v_ in r[1].items()}, ensure_ascii=False, indent=True)}"}
                for r in results]


class DetectionVisualize(TaskVisualize):
    TRAINING = ['loss', 'loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']

    def __init__(self, model_root, single: bool = False, save_dir: str = './'):
        super().__init__(model_root, single, task_type='where', save_dir=save_dir)
        self.results = {k_: pd.read_csv(v_.training_log, header=0, index_col=0)[DetectionVisualize.TRAINING]
                        for k_, v_ in self.sub_viz.items()}

    def viz(self):
        viz_result = []
        # plot training log
        for k, v in self.results.items():
            viz_spec = {'model_name': f"where.{k}", "result_list": []}
            
            v.plot(figsize=(16, 9))
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(self.save_dir, k, f"loss.svg"), dpi=300)
            plt.close()
            viz_spec['result_list'].append({'img': os.path.join(self.save_dir, k, f"loss.svg"), 'desc': f"loss曲线"})
            # 保存valid的数据曲线
            sub_viz = self.sub_viz[k]
            df = pd.read_csv(sub_viz._results, header=0, index_col=0)
            
            df.plot(figsize=(16, 9))
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(self.save_dir, k, f"metric.svg"), dpi=300)
            plt.close()
            viz_spec['result_list'].append({'img': os.path.join(self.save_dir, k, f"metric.svg"),
                                            'desc': f"准确率-召回率结果"})
            viz_result.append(viz_spec)
        return viz_result

    def predict(self, sample, model_name):
        assert model_name in self.model
        if self.model[model_name] is None:
            self.model[model_name] = init_obj(self.sub_viz[model_name].sub_viz_dir)
        results = inference_obj(sample, *self.model[model_name], save_dir=os.path.join(self.save_dir, model_name))
        return [{'img': res_[1],
                 'desc': json.dumps(res_[2], ensure_ascii=False, indent=True)} for res_ in results]


class SegmentationVisualize(TaskVisualize):
    TRAINING = ['loss']
    VALID = ['global_acc', 'acc_per_class', 'iou_per_class', 'mIoU']

    def __init__(self, model_root, single: bool = False, save_dir: str = './'):
        super().__init__(model_root, single, task_type='which', save_dir=save_dir)
        self.results = {k_: pd.read_csv(v_.training_log, header=0, index_col=0)[SegmentationVisualize.TRAINING]
                        for k_, v_ in self.sub_viz.items()}

    def viz(self):
        viz_result = []
        # plot training log
        for k, v in self.results.items():
            viz_spec = {'model_name': f"which.{k}", "result_list": []}
            for name in SegmentationVisualize.TRAINING:
                
                v[name].plot(subplots=True, figsize=(16, 9))
                plt.legend(loc='lower right')
                plt.savefig(os.path.join(self.save_dir, k, f"{name}.svg"), dpi=300)
                viz_spec['result_list'].append({'img': os.path.join(self.save_dir, k, f"{name}.svg"),
                                                'desc': f"{name}曲线"})
                plt.close()
            # 保存valid的数据曲线
            sub_viz = self.sub_viz[k]
            sub_valid_dir = os.path.join(self.sub_viz[k].sub_viz_dir, '../valid')
            sub_valid_dir = [os.path.join(sub_valid_dir, l_) for l_ in os.listdir(sub_valid_dir)
                             if l_.endswith('.txt')]
            valid = [json.loads(open(svd, encoding='utf8').read())
                     for svd in sorted(sub_valid_dir, key=lambda x: int(os.path.basename(x).split('-')[1][:-4]))]
            header = []
            for v_ in SegmentationVisualize.VALID:
                if 'per_class' in v_:
                    header.extend([f"{v_.replace('per_class', l_)}" for l_ in sub_viz.labels])
                else:
                    header.append(v_)
            data = []
            for val in valid:
                epoch_valid_data = []
                for v_ in SegmentationVisualize.VALID:
                    if isinstance(val[v_], (list, tuple)):
                        epoch_valid_data.extend(list(val[v_]))
                    else:
                        epoch_valid_data.append(val[v_])
                data.append(epoch_valid_data)
            
            df = pd.DataFrame(data, columns=header, index=[f"Epoch-{idx}" for idx in range(len(data))])
            df.plot(figsize=(16, 9))
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(self.save_dir, k, f"metric.svg"), dpi=300)
            plt.close()
            # content = open(sub_viz._results, encoding='utf8').read()
            content = ''
            viz_spec['result_list'].append({'img': os.path.join(self.save_dir, k, f"metric.svg"),
                                            'desc': f"准确率细分曲线 {content}"})
            viz_result.append(viz_spec)
        return viz_result

    def predict(self, sample, model_name):
        assert model_name in self.model
        if self.model[model_name] is None:
            self.model[model_name] = init_seg(self.sub_viz[model_name].sub_viz_dir)
        results = inference_seg(sample, *self.model[model_name], save_dir=os.path.join(self.save_dir, model_name))
        return [{"img": res_, 'desc': "分割结果"} for res_ in results]


if __name__ == '__main__':
    # cv = ClassificationVisualize('../../classification/20211014')
    # cv.viz()
    # res = cv.predict([r'G:\skin_classification\images\ISIC_0000012.jpg'], 'resnet18')
    # print(res)

    # dv = DetectionVisualize('../../detection/20211017')
    # viz_results = dv.viz()
    # print(viz_results)
    # res = dv.predict([r'/Users/zhangzhiwei/projects/python/covid-chestxray-dataset/images/0a7faa2a.jpg'],
    #                  'fasterrcnn_resnet50_fpn')
    # print(res)

    sv = SegmentationVisualize('/Users/zhangzhiwei/Downloads/models', save_dir='/Users/zhangzhiwei/Downloads/models')
    viz_results = sv.viz()
    print(viz_results)
    # res = sv.predict([r'/Users/zhangzhiwei/projects/python/covid-chestxray-dataset/images/0a7faa2a.jpg'],
    #                  'fcn_resnet50')
    # print(res)
