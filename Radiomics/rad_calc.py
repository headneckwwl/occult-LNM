
import json
import os
import tempfile

import joblib
import pandas as pd
from flask import request, Flask, Response, jsonify

from _algo.custom.components.Radiology import ConventionalRadiomics
from _algo.custom.components.ml import inference as ml_inference
from _algo.custom.components.ml import transform as ml_transform

app = Flask(__name__, static_folder='html')
radio_exec = ConventionalRadiomics(param_file=None, correctMask=True)
sel_features = ['original_firstorder_TotalEnergy', 'original_ngtdm_Strength',
                'original_shape_Maximum2DDiameterColumn',
                'original_shape_MinorAxisLength', 'Age']
trans = json.loads(open('task_settings/rad/transform.json').read())
model = joblib.load(r'C:\Tasks\BaiduSyncdisk\TaskSpec51-YangCun\models/MLP_label.pkl')


@app.route('/model/test', methods=['POST'])
def modelTest():
    modelName = request.form['modelName']
    f = request.files['file']
    pic1 = {'img': '/photo/pid', 'desc': 'auc曲线'}
    pic2 = {'img': '/photo/pid', 'desc': '混淆矩阵'}
    result_list = []
    result_list.append(pic1)
    result_list.append(pic2)
    return jsonify(result_list)


@app.route('/resource', methods=['GET'])
def resource():
    return app.send_static_file(request.args.get('name'))


@app.route('/photo/pid')
def get_frame():
    # 图片上传保存的路径
    with open(r'timg.jpeg', 'rb') as f:
        image = f.read()
        resp = Response(image, mimetype='image/png')
        return resp


@app.route('/nocode/home', methods=['GET'])
def testPage():
    return app.send_static_file('nocode.html')


@app.route('/nocode/config', methods=['GET'])
def configGet():
    with open('task_settings/rad/config.json', encoding='utf-8-sig') as f:
        content = json.loads(f.read())
    return jsonify(content)


@app.route('/nocode/submit', methods=['POST'])
def nocodeSubmit():
    age = int(request.form['age'])
    sex = request.form['sex']
    # floatTest = request.form['floatTest']
    # intTest = request.form['intTest']
    image1 = request.files['image1']
    image2 = request.files['image2']
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = os.path.join(tmp_dir, image1.filename)
        mask_path = os.path.join(tmp_dir, image2.filename)
        image1.save(image_path)
        image2.save(mask_path)

        radio_exec.extract([image_path], [mask_path])
        rad_features = radio_exec.get_label_data_frame()
        clinic_features = pd.DataFrame([[image1.filename, age, sex]], columns=['ID', 'Age', 'sex'])
        features = pd.merge(rad_features, clinic_features, on='ID', how='inner')
        print(features)
        sample = ml_transform(features, transform=trans, sel_featues=sel_features)
        sample = {image1.filename: sample}
        results = ml_inference(model, sample)
        result_list = [{"text": f"{results}"}]
    return jsonify(result_list)


if __name__ == '__main__':
    app.run('127.0.0.1', port=5000, debug=True)
