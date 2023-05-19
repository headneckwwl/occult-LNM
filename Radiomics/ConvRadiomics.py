
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier, LGBMRegressor
from pandas import Index
from radiomics import featureextractor
from scipy.stats import pearsonr
from skimage.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

logger = logging.root

DEFAULT_CLF_SETTINGS = {
    'svm': {},
    'knn': {'algorithm': 'kd_tree'},
    'dt': {'max_depth': None, 'min_samples_split': 2, 'random_state': 0},
    'rf': {'n_estimators': 10, 'max_depth': None, 'min_samples_split': 2, 'random_state': 0},
    'et': {'n_estimators': 10, 'max_depth': None, 'min_samples_split': 2, 'random_state': 0},
    'xgb': {'n_estimators': 50, 'max_depth': 5, 'objective': 'binary:logistic', 'use_label_encoder': False},
    'lgb': {'n_estimators': 80, 'max_depth': 4, 'objective': 'binary'}
}

DEFAULT_REG_SETTINGS = {
    'svm': {},
    'lasso': {'alpha': 0.1},
    'knn': {'algorithm': 'kd_tree'},
    'dt': {'max_depth': None, 'min_samples_split': 2, 'random_state': 0},
    'rf': {'n_estimators': 10, 'max_depth': None, 'min_samples_split': 2, 'random_state': 0},
    'et': {'n_estimators': 10, 'max_depth': None, 'min_samples_split': 2, 'random_state': 0},
    'xgb': {'n_estimators': 50, 'max_depth': 5, 'objective': 'reg:squarederror', 'use_label_encoder': False},
    'lgb': {'n_estimators': 80, 'max_depth': 4, 'objective': 'regression'}
}


class ConventionalRadiomics(object):
    def __init__(self, params_file: str = None, **params):
        settings = {}
        if params_file is not None and os.path.exists(params_file):
            _, ext = os.path.splitext(params_file)
            with open(params_file) as pf:
                if ext.lower() == 'json':
                    settings = json.loads(pf.read())
                elif ext.lower() == 'yaml':
                    settings = yaml.load(pf.read(), Loader=yaml.FullLoader)
                else:
                    raise ValueError(f"Parameters file {params_file}'s format({ext}) not found!")
        settings.update(params)
        self.settings = settings
        self._features = {}
        self.feature_names = set()
        self.extractor = None
        self.df = None

        # Initialize feature extractor
        self.init_extractor(self.settings)

    def init_extractor(self, settings=None):
        settings = settings or self.settings
        self.extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    def extract(self, images: Union[str, List[str]], masks: Union[str, List[str]],
                labels: Union[int, List[int]] = 1, settings=None):
        if settings is not None:
            self.init_extractor(settings)
        if not isinstance(images, (list, tuple)):
            images = [images]
        if not isinstance(masks, (list, tuple)):
            masks = [masks]
        assert len(images) == len(masks), '图像和标注数据必须一一对应。'
        if not isinstance(labels, (list, tuple)):
            labels = [labels]
        for image, mask in zip(images, masks):
            print(f'\t Extracting {image}...')
            image_name = os.path.basename(image)
            self._features[image_name] = {}
            statics = {}
            features = {}
            for label in labels:
                featureVector = self.extractor.execute(image, mask, label=label)
                for featureName in featureVector.keys():
                    f_type, c_name, f_name = featureName.split('_')
                    if f_type == 'diagnostics':
                        if c_name not in statics:
                            statics[c_name] = {}
                        statics[c_name].update({f_name: featureVector[featureName]})
                    elif f_type == 'original':
                        self.feature_names.add(f"{c_name}_{f_name}")
                        if c_name not in features:
                            features[c_name] = {}
                        features[c_name].update({f_name: float(featureVector[featureName])})
                self._features[image_name][label] = {"statics": statics, 'features': features}
        # print(json.dumps(self._features, indent=True))
        return self._features

    @property
    def features(self, labels: Union[list, tuple, set] = None):
        if self._features:
            feature = {}
            for k_, v_ in self._features.items():
                feature[k_] = {l_: f_['features'] for l_, f_ in v_.items() if labels is None or l_ in labels}
            return feature
        else:
            logger.warning(f'No features found! Perhaps you should input images and masks!')

    @property
    def statics(self, labels: Union[list, tuple, set] = None):
        if self._features:
            statics = {}
            for k_, v_ in self._features.items():
                statics[k_] = {l_: f_['features'] for l_, f_ in v_.items() if labels is None or l_ in labels}
            return statics
        else:
            logger.warning(f'No features found! Perhaps you should input images and masks!')

    def get_label_data_frame(self, label: int = 1, column_names=None):
        column_names = column_names or sorted(list(self.feature_names))
        indexes = []
        df = []
        for k_, v_ in self.features.items():
            indexes.append(k_)
            data = []
            if label in v_:
                for name in column_names:
                    c_name, f_name = name.split('_')
                    data.append(v_[label][c_name][f_name])
                df.append(data)
        self.df = pd.DataFrame(df, columns=column_names, index=indexes)
        return self.df


class Analyser(object):
    def __init__(self, features: Union[pd.DataFrame, np.ndarray], task_type, labels=None,
                 compress_dim: int = None, n_clusters: int = None, settings: dict = None):
        n_samples, n_features = features.shape
        self.features = features if isinstance(features, pd.DataFrame) else pd.DataFrame(features)
        if isinstance(self.features.columns, Index):
            self.feature_names = self.features.columns
        else:
            self.feature_names = [f"X{idx + 1}" for idx in range(n_features)]
            self.features.columns = self.feature_names
        self.labels = np.array(labels)
        self.compressor = None
        self.clusterer = None
        self.classifier = []
        self.regressor = []
        assert task_type in ['clf', 'reg', 'cls'], \
            f"任务必须是分类（clf）、回归（reg）、无监督聚类（cls），但现在是{task_type}。"

        if compress_dim is not None:
            self.init_compressor(compress_dim)
        assert task_type != 'cls' or n_clusters is not None, '选定的任务无监督聚类任务，但是聚类个数未指定。'
        if n_clusters is not None:
            self.init_cluster(n_clusters)

        # Bind task type
        if task_type == 'clf':
            settings = settings or DEFAULT_CLF_SETTINGS
            self.labels = self.labels.astype(np.int)
            self.init_classifier(**settings)
        if task_type == 'reg':
            settings = settings or DEFAULT_REG_SETTINGS
            self.init_regressor(**settings)
        self.task_type = task_type

        self.model = self.classifier or self.regressor
        if task_type in ['clf', 'reg'] and not self.model:
            raise ValueError(f"在这些配置里面，没有找到一个可以配置的模型。\n{json.dumps(settings, indent=True)}")
        self.predictions = []
        self.metrics = []
        self.bst_model = None
        self.clusters = []
        self.use_compress = False

    def init_regressor(self, **kwargs):
        if 'svm' in kwargs:
            self.regressor.append(('SVM', SVR(**kwargs['svm'])))
        if 'lasso' in kwargs:
            self.regressor.append(('Lasso', Lasso(**kwargs['lasso'])))
        if 'knn' in kwargs:
            self.regressor.append(('KNN', KNeighborsRegressor(**kwargs['knn'])))
        if 'rf' in kwargs:
            self.regressor.append(('RandomForest', RandomForestRegressor(**kwargs['rf'])))
        if 'et' in kwargs:
            self.regressor.append(('ExtraTrees', ExtraTreesRegressor(**kwargs['et'])))
        if 'dt' in kwargs:
            self.regressor.append(('DecisionTree', DecisionTreeRegressor(**kwargs['dt'])))
        if 'xgb' in kwargs:
            self.regressor.append(('XGBoost', XGBRegressor(**kwargs['xgb'])))
        if 'lgb' in kwargs:
            self.regressor.append(('LightGBM', LGBMRegressor(**kwargs['lgb'])))

    def init_classifier(self, **kwargs):
        if 'svm' in kwargs:
            self.classifier.append(('SVM', SVC(**kwargs['svm'])))
        if 'knn' in kwargs:
            self.classifier.append(('KNN', KNeighborsClassifier(**kwargs['knn'])))
        if 'rf' in kwargs:
            self.classifier.append(('RandomForest', RandomForestClassifier(**kwargs['rf'])))
        if 'et' in kwargs:
            self.classifier.append(('ExtraTrees', ExtraTreesClassifier(**kwargs['et'])))
        if 'dt' in kwargs:
            self.classifier.append(('DecisionTree', DecisionTreeClassifier(**kwargs['dt'])))
        if 'xgb' in kwargs:
            self.classifier.append(('XGBoost', XGBClassifier(**kwargs['xgb'])))
        if 'lgb' in kwargs:
            self.classifier.append(('LightGBM', LGBMClassifier(**kwargs['lgb'])))

    def init_compressor(self, dim=None):
        if isinstance(dim, int):
            self.compressor = PCA(dim)
        else:
            self.compressor = None

    def init_cluster(self, n_cluster=None, **kwargs):
        if isinstance(n_cluster, int):
            self.clusterer = KMeans(n_clusters=n_cluster, random_state=0, **kwargs)
        else:
            self.clusterer = None

    def cmp_feature(self, dim=2):
        assert dim in [2, 3]
        pca = PCA(dim)
        return pca.fit_transform(self.features)

    def train(self, use_compress: bool = False, test_size: float = 0.2):
        if use_compress and self.compressor is None:
            logger.warning(f"压缩器没有初始化，你可以需要指定压缩特征的维度，我们将使用原始特征替代！")
        features = self.features if not use_compress or self.compressor is None \
            else self.compressor.fit_transform(self.features)
        self.use_compress = use_compress
        if self.task_type == 'cls':
            index = self.features.index
            self.clusterer.fit(self.features)
            clusters = self.clusterer.labels_.tolist()
            feature_2dim = self.cmp_feature().tolist()
            feature_2dim = feature_2dim / np.max(np.abs(feature_2dim)) * 10
            x_, y_ = zip(*feature_2dim)
            self.clusters = zip(index, clusters, x_, y_)
            return self.clusters
        else:
            X_train, X_test, y_train, y_test = train_test_split(features, self.labels,
                                                                random_state=19901005, test_size=test_size)
            # Predict each model in test data set.
            for model_name, model in self.model:
                model.fit(X_train, y_train)
                if model_name.lower() == 'lasso':
                    lasso_str = [f"{model.intercept_}"]
                    for fea_name, fea_weight in zip(self.feature_names, model.coef_):
                        lasso_str.append(f"({fea_weight}*{fea_name})")
                    print(' + '.join(lasso_str))
                self.predictions.append((model_name, y_test, model.predict(X_test)))

            # Evaluate each model
            if self.task_type == 'clf':
                score = [(model_name, accuracy_score(y_test, pred)) for model_name, y_test, pred in self.predictions]
                bst = 0
                for idx, (model_name, s) in enumerate(score):
                    if s > bst:
                        self.bst_model = (model_name, self.model[idx])
                        bst = s
            else:
                score = [(model_name, mean_squared_error(y_test, pred)) for model_name, y_test, pred in
                         self.predictions]
                bst = sys.maxsize
                for idx, (model_name, s) in enumerate(score):
                    if s < bst:
                        self.bst_model = (model_name, self.model[idx])
                        bst = s

            # Get best model
            self.metrics = score
            return self.metrics

    def inference(self, samples):
        if self.use_compress:
            samples = self.compressor.transform(samples)
        if self.task_type == 'cls':
            return self.clusterer.predict(samples).tolist()
        else:
            return self.bst_model.predict(samples)

    def statistics(self):
        cov, pvalue = self.__calc_covariance()
        return cov, pvalue, self.features.describe()

    def __calc_covariance(self):
        fea = self.features.to_numpy()
        _, n_features = fea.shape
        cov_matrix = np.zeros((n_features, n_features))
        pvalue_matrix = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(i, n_features):
                p_co, p_va = pearsonr(fea[:, i], fea[:, j])
                cov_matrix[i, j] = p_co
                cov_matrix[j, i] = p_co
                pvalue_matrix[i, j] = p_va
                pvalue_matrix[j, i] = p_va

        return pd.DataFrame(cov_matrix, index=self.feature_names, columns=self.feature_names), \
               pd.DataFrame(pvalue_matrix, index=self.feature_names, columns=self.feature_names)


def get_image_mask_from_dir(path, limit: int = None):
    items = os.listdir(path)
    assert 'images' in items and 'masks' in items
    images_path = Path(os.path.join(path, 'images'))
    masks_path = Path(os.path.join(path, 'masks'))

    images = []
    masks = []
    for l_ in os.listdir(images_path):
        if not l_.startswith('.'):
            f_name, _ = os.path.splitext(l_)
            mask_file = list(masks_path.glob(f_name + '*'))
            if len(mask_file) == 1:
                images.append(os.path.abspath(os.path.join(images_path, l_)))
                masks.append(os.path.abspath(mask_file[0]))
    return images[:limit], masks[:limit]


def test_cr():
    image_root = os.path.expanduser('~/Downloads/pre')
    images = sorted([os.path.join(image_root, l) for l in os.listdir(image_root) if l.endswith('.gz')])
    imageName = images[:len(images) // 2]
    maskName = images[len(images) // 2:]
    print(imageName, maskName)
    cr = ConventionalRadiomics()
    cr.extract(imageName, maskName, labels=[1, 2])
    # print(json.dumps(cr.features, indent=True))
    print(cr.get_label_data_frame(2))


def test_analyser_clf():
    bc_data = pd.read_csv('~/bc_data.csv', header=0)
    data = bc_data.drop(['id'], axis=1)
    X_data = data.drop(['diagnosis'], axis=1)
    y_data = np.ravel(data[['diagnosis']].applymap(lambda x: 0 if x == 'M' else 1))
    analyser = Analyser(X_data, labels=y_data, settings=DEFAULT_CLF_SETTINGS, task_type='clf')
    print(analyser.statistics())
    analyser.train()
    print(analyser.metrics)


def test_analyser_reg():
    bc_data = pd.read_csv('~/bc_data.csv', header=0)
    data = bc_data.drop(['id'], axis=1)
    X_data = data.drop(['diagnosis'], axis=1)
    y_data = np.ravel(data[['diagnosis']].applymap(lambda x: 0 if x == 'M' else 1))

    analyser = Analyser(X_data, labels=y_data, settings=DEFAULT_REG_SETTINGS, task_type='reg')
    print(analyser.statistics())
    analyser.train()
    print(analyser.metrics)


def test_analyser_cls():
    bc_data = pd.read_csv('bc_data.csv', header=0)
    data = bc_data.drop(['id'], axis=1)
    X_data = data.drop(['diagnosis'], axis=1)
    y_data = np.ravel(data[['diagnosis']].applymap(lambda x: 0 if x == 'M' else 1))

    analyser = Analyser(X_data, settings=DEFAULT_REG_SETTINGS, task_type='cls', n_clusters=5, compress_dim=30)
    # print(analyser.statistics())
    analyser.train(use_compress=True)
    print(pd.DataFrame(analyser.clusters))


if __name__ == "__main__":
    test_analyser_cls()
    # print(get_image_mask_from_dir(os.path.expanduser('~/Downloads/pre')))
