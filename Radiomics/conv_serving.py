# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/10/29
# Blog: www.wangqingbaidu.cn
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.
import json
import os
import traceback

import pandas as pd
from flask import Flask, make_response, request, jsonify

from ConvRadiomics import ConventionalRadiomics, Analyser, get_image_mask_from_dir, logger

app = Flask(__name__, static_folder='html')

conv_radiomics: ConventionalRadiomics
IMAGE_STATIC = 'static'
os.makedirs(IMAGE_STATIC, exist_ok=True)


@app.route('/', methods=['GET'])
def home_page():
    print(os.getcwd())
    return app.send_static_file('echart.html')


def form_matrix(data: pd.DataFrame):
    x_axis = [i for i in data.columns]
    y_axis = [i for i in data.index]
    return {'x': x_axis, 'y': y_axis, 'data': data.to_numpy().tolist()}


@app.route('/feature/get', methods=['POST'])
def extract_feature():
    try:
        global conv_radiomics
        model_root = request.json['dir']
        label = request.form.get('label', 1)
        images, masks = get_image_mask_from_dir(model_root)
        params = request.files.get('file', None)
        params_file = None
        if params:
            params_file = os.path.join(IMAGE_STATIC, 'extract_params.txt')
            params.save(params_file)
        conv_radiomics = ConventionalRadiomics(params_file)

        if images and masks:
            print(f'Start extracting feature from {model_root}')
            conv_radiomics.extract(images, masks, label)
            feature = conv_radiomics.get_label_data_frame(label)
            # with open('data.json', 'w') as fi:
            #     print(json.dumps(form_matrix(feature.T)), file=fi)
            return jsonify(form_matrix(feature.T))
        else:
            return make_response('目录不存在或者没有找到匹配的样本，请检查设置！', 500)
    except Exception as e:
        traceback.print_exc()
        return make_response(str(e), 500)


@app.route('/analyze/get', methods=['POST'])
def analysis():
    try:
        global conv_radiomics
        dim = request.form.get('dim', None)
        n_clusters = request.form.get('n_clusters', 4)
        analyser = Analyser(conv_radiomics.df, task_type='cls', compress_dim=dim, n_clusters=n_clusters)
        cov, pvalue, desc = analyser.statistics()
        clusters = analyser.train(use_compress=True if dim is not None else False)
        return jsonify({"cov": form_matrix(cov),
                        'pvalue': form_matrix(pvalue),
                        'desc': form_matrix(desc),
                        'cluster': list(clusters)})
    except Exception as e:
        traceback.print_exc()
        return make_response(str(e), 500)


@app.route('/model_results', methods=['POST'])
def model_run():
    try:
        global conv_radiomics
        task_type = request.form.get('task', 'reg')
        settings = json.loads(request.form.get('settings', 'null'))

        sample_names = conv_radiomics.features.keys()
        labels = request.files.get('file', None)
        if labels:
            data = {}
            labels_file = os.path.join(IMAGE_STATIC, 'labels.txt')
            labels.save(labels_file)
            with open(labels_file) as f:
                for l in f.readlines():
                    k, v = l.strip().split()
                    data[k] = float(v)
            y_data = [data[s] for s in sample_names]
        else:
            raise ValueError("没有文件")

        analyser = Analyser(conv_radiomics.df, labels=y_data, settings=settings, task_type=task_type)
        analyser.train()
        return jsonify(analyser.metrics)
    except Exception as e:
        traceback.print_exc()
        return make_response(str(e), 500)


@app.route('/resource', methods=['GET'])
def resource():
    return app.send_static_file(request.args.get('name'))


if __name__ == '__main__':
    os.makedirs(IMAGE_STATIC, exist_ok=True)
    app.run('127.0.0.1', port=5000, debug=False)
