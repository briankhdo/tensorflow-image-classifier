#!/usr/bin/python3
import re
from io import BytesIO
import decimal
import flask.json
from flask import Flask, send_file, request, jsonify, render_template, send_from_directory, redirect
from PIL import Image, ImageDraw, ExifTags, ImageFont
import requests
import numpy as np
import tensorflow as tf
import tensorflow as tf_classify
import os, sys, time
import json
import uuid
import base64
import copy
import glob
import redis

import cv2

from stat import S_ISREG, ST_CTIME, ST_MODE

from align_dlib import AlignDlib

classify_redis = redis.Redis(host='localhost', port=6379, db=0)

config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 1

CKPT_MODEL_NAME = ['Mar 15', 'Mar 16']
CKPT_PATHS = ['./tf_files/retrained_graph.pb', './tf_files_mar16/retrained_graph.pb']

crop_dim = 180

LABEL_PATHS = ['./tf_files/retrained_labels.txt', './tf_files_mar16/retrained_labels.txt']

align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))

app = Flask(__name__)

def image2array(image):
    (w, h) = image.size
    return np.array(image.getdata()).reshape((h, w, 3)).astype(np.uint8)

def array2image(arr):
    return Image.fromarray(np.uint8(arr))

classification_graphs = []
prediction_sesses = []
softmax_tensors = []
for ckpt_path in CKPT_PATHS:
    classification_graph = tf.Graph()
    with classification_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(ckpt_path, 'rb') as fid:
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name='')
    classification_graphs.append(classification_graph)

    prediction_sess = None
    softmax_tensor = None
    with classification_graph.as_default():
        with tf.Session(config=config,graph=classification_graph) as sess:
            prediction_sess = sess
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    prediction_sesses.append(prediction_sess)
    softmax_tensors.append(softmax_tensor)

print('Model Loaded')

def detect_faces(image):
    pil_image = image.convert('RGB')
    open_cv_image = np.array(pil_image)
    '''Plots the object detection result for a given image.'''
    bbes = align_dlib.getFaceBoundingBoxes(open_cv_image)

    aligned_images = []
    if bbes is not None:
        for bb in bbes:
            aligned = align_dlib.align(crop_dim, open_cv_image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
            if aligned is not None:
                aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                image_data = cv2.imencode('.jpg', aligned)[1].tostring()
                aligned_images.append(image_data)
    return aligned_images, bbes

label_lines = []
for PATH_TO_IMAGE_LABELS in LABEL_PATHS:
    label_lines.append([line.rstrip() for line 
                   in tf.gfile.GFile(PATH_TO_IMAGE_LABELS)])


def classify_faces(faces, classification_graph, label_lines):
    prediction_sess = None
    softmax_tensor = None
    results = []
    with classification_graph.as_default():
        with tf.Session(config=config,graph=classification_graph) as sess:
            prediction_sess = sess
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            for image in faces:
                predictions = prediction_sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image})
                # gibt prediction values in array zuerueck:
                top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                print(top_k)
                result = []
                max_score = 0
                max_human_string = ''
                for node_id in top_k:
                    human_string = label_lines[node_id].replace("user checkin ", "")
                    score = predictions[0][node_id]
                    if max_score < score:
                        max_score = score
                        max_human_string = human_string
                    if score >= 0.2:
                        data = [human_string, score*100.0]
                        result.append(data)
                if len(result) == 0:
                    data = [max_human_string, max_score*100.0]
                    result.append(data)
                results.append(result)
    return results

@app.route('/assets/<path:path>')
def send_assets(path):
    return send_from_directory('assets', path)

@app.route('/images/<path:path>')
def send_js(path):
    return send_from_directory('images', path)

@app.route('/learning/<path:path>')
def learning(path):
    return send_from_directory('learning', path)

@app.route('/')
def root():
   return render_template('upload.html')

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/classify')
def classify():

    dirpath = './learning'
    all_files = glob.glob(dirpath + '/*')
    all_files.sort()

    user_id = request.args.get('user_id')
    if user_id:
        dirpath = './learning/user_checkin_' + user_id
        files = glob.glob(dirpath + '/*')
        files.sort()
        next_user = request.args.get('next')
        current_user_index = all_files.index("./learning/user_checkin_" + user_id)
        if current_user_index + 1 < len(all_files):
            next_user = os.path.basename(all_files[current_user_index + 1]).replace("user_checkin_", "")

        if next_user is None:
            next_user = ''

        images = []
        for index, path in enumerate(files):
            redis_key_name = 'user_' + user_id + ':' + os.path.basename(path)
            classified = classify_redis.get(redis_key_name)
            if classified:
                classified = classified.decode('utf-8')
            else:
                classified = 'correct'
            images.append({
                'path': path[1:],
                'name': os.path.basename(path),
                'classify': classified
            })

        return render_template('classify_user.html', images=images, user=user_id, next_user=next_user)
    else:

        files = all_files

        users = []
        for index, path in enumerate(files):
            if 'user_checkin_' in os.path.basename(path):
                classify_done = True
                username = os.path.basename(path).replace('user_checkin_', '')
                last_classify = classify_redis.get("user_" + username + ":last_classify")
                last_updated = classify_redis.get("user_" + username + ":last_updated")

                current_time = time.time()

                if last_classify:
                    last_classify = float(last_classify)
                else:
                    last_classify = 0

                if last_updated:
                    last_updated = float(last_updated)
                else:
                    classify_redis.set("user_" + username + ":last_updated", current_time)
                    last_updated = current_time

                if last_classify < last_updated:
                    classify_done = False


                next_user = ''
                if index + 1 < len(files):
                    next_user = os.path.basename(files[index+1]).replace('user_checkin_', '')

                user = {
                    'name': username,
                    'time': os.path.getctime(path),
                    'done': classify_done,
                    'last_updated': last_updated,
                    'last_classify': last_classify,
                    'next': next_user
                }
                users.append(user)


        return render_template('classify_listing.html', users=users)
    

@app.route('/save_classification', methods=['POST'])
def save_classification():
    user_id = request.form.get('user_id')

    images = {}
    for key in request.form:
        if key != 'user_id' and key != 'submit':
            value = request.form.get(key)
            print(value)
            images[key] = value
            print("user_" + user_id + ":" + key)
            classify_redis.set("user_" + user_id + ":" + key.replace("image-", ''), value)

    classify_redis.set('user_' + user_id + ':last_classify', time.time())

    if request.form.get('submit') != '':
        return redirect('./classify?user_id=' + request.form.get('submit'))
    else:
        return redirect('./classify')

@app.route('/recognize', methods=['POST'])
def recognize():
    file = request.files['image']
    if file:
        data = file.read()
        org_image = Image.open(BytesIO(data))
        if hasattr(org_image, '_getexif'):
            for orientation in ExifTags.TAGS.keys() : 
                if ExifTags.TAGS[orientation]=='Orientation' : break
            if org_image._getexif() != None:
                exif=dict(org_image._getexif().items())
                if orientation in exif:
                    if   exif[orientation] == 3 : 
                        org_image=org_image.rotate(180, expand=True)
                    elif exif[orientation] == 6 : 
                        org_image=org_image.rotate(270, expand=True)
                    elif exif[orientation] == 8 : 
                        org_image=org_image.rotate(90, expand=True)

        size = 800,800
        org_image.thumbnail(size,Image.ANTIALIAS)
        
        faces, boxes = detect_faces(org_image)

        images = []

        detections = []

        detection_data = []

        for index, model_name in enumerate(CKPT_MODEL_NAME):

            classification_graph = classification_graphs[index]
            classify_results = classify_faces(faces, classification_graph, label_lines[index])
            detections.append(classify_results)

            image = copy.copy(org_image)

            draw = ImageDraw.Draw(image)

            # select scores
            result = []

            if boxes is not None:

                for index, box in enumerate(boxes):
                    area = [
                        box.left(),
                        box.top(),
                        box.right(),
                        box.bottom()
                    ]
                    face_classes = classify_results[index]
                    class_name = 'unknown'
                    img_fraction = 1.0

                    posibility = 1
                    if len(face_classes) > 0:
                        possible_class = face_classes[0]
                        class_name = possible_class[0]
                        posibility = possible_class[1]

                    draw.rectangle(area, fill=None, outline=(0,255,0,120))
                    left = box.left()
                    if left < 0:
                        left = 0
                    draw.text([(left, box.top() - 20)],
                        class_name, 
                        fill=(0,255,0,120))
                    draw.text([(left, box.top() - 10)],
                        str(posibility), 
                        fill=(0,255,0,120))

            else:
                draw.text([10, 10],
                    'Cannot detect faces', 
                    fill=(0,255,0,120))

            byte_io = BytesIO()
            file_name = "./images/%s_result.jpg" % uuid.uuid4().hex
            image.save(file_name, 'JPEG')
            images.append(file_name)

            detection = {
                "image": file_name,
                "detections": classify_results,
                "model_name": model_name
            }

            detection_data.append(detection)

        return render_template('upload.html', detection_data=detection_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)

