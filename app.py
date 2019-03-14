#!/usr/bin/python3
import re
from io import BytesIO
import decimal
import flask.json
from flask import Flask, send_file, request, jsonify, render_template
from PIL import Image, ImageDraw, ExifTags, ImageFont
import requests
import numpy as np
import tensorflow as tf
import tensorflow as tf_classify
import os
import json
import uuid

import cv2

from align_dlib import AlignDlib

config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 1

PATH_TO_IMAGE_CKPT = './tf_files/retrained_graph.pb'

crop_dim = 180

PATH_TO_IMAGE_LABELS = './tf_files/retrained_labels.txt'

align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))

app = Flask(__name__)

def image2array(image):
    (w, h) = image.size
    return np.array(image.getdata()).reshape((h, w, 3)).astype(np.uint8)

def array2image(arr):
    return Image.fromarray(np.uint8(arr))

classification_graph = tf.Graph()
with classification_graph.as_default():
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_IMAGE_CKPT, 'rb') as fid:
        graph_def.ParseFromString(fid.read())
        tf.import_graph_def(graph_def, name='')

prediction_sess = None
softmax_tensor = None
with classification_graph.as_default():
    with tf.Session(config=config,graph=classification_graph) as sess:
        prediction_sess = sess
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

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

label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile(PATH_TO_IMAGE_LABELS)]


def classify_faces(faces):
    results = []
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

@app.route('/results/json')
def results_json():
    file = request.args.get('job_id', "")
    if os.path.isdir(file):
        return send_file(file + "/result.json", mimetype='application/json')
    else:
        return jsonify(
            code=404,
            message="Invalid job_id"
        )

@app.route('/')
def index():
   return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
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

        classify_results = classify_faces(faces)

        draw = ImageDraw.Draw(org_image)

        # select scores
        result = []
        # folder_name = uuid.uuid4().hex
        # if os.path.isdir(folder_name) == False:
        #     os.mkdir(folder_name)

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

                fontsize = 2
                box_size = box.width()
                while font.getsize(class_name)[0] < box_size*img_fraction:
                    # iterate until the text size is just larger than the criteria
                    fontsize += 1

                # optionally de-increment to be sure it is less than criteria
                fontsize -= 1
                font = ImageFont.truetype("arial.ttf", fontsize)

                posibility = 1
                if len(face_classes) > 0:
                    possible_class = face_classes[0]
                    class_name = possible_class[0]
                    posibility = possible_class[1]

                draw.rectangle(area, fill=None, outline=(0,255,0,120))
                draw.text([(box.left(), box.top() - 20)],
                    class_name, 
                    fill=(0,255,0,120), font=font)
                draw.text([(box.left(), box.top() - 10)],
                    str(posibility), 
                    fill=(0,255,0,120))

        else:
            draw.text([10, 10],
                'Cannot detect faces', 
                fill=(0,255,0,120))

        byte_io = BytesIO()
        org_image.save(byte_io, 'JPEG')
        byte_io.seek(0)

        return send_file(byte_io, mimetype='image/jpeg')


        # boxes = image[0].tolist()
        # scores = image[1].tolist()
        # products = {}
        # brands = {}
        # brands["unsure"] = 0

        # for i in range(0, len(scores) - 1):
        #     if scores[i] > 0.8:
        #         width, height = org_image.size
        #         box = boxes[i]
        #         area = [
        #             int(box[1] * width),
        #             int(box[0] * height),
        #             int(box[3] * width),
        #             int(box[2] * height)]
        #         # crop image
        #         out_path = folder_name + "/CROP_" + str(i) + ".JPG"
        #         org_image.crop(area).save(out_path, 'JPEG')
        #         cropped_img = tf.gfile.FastGFile(out_path, 'rb').read()
        #         detections = classify_object(cropped_img)

        #         result.append({
        #             'box': area,
        #             'score': scores[i],
        #             'name': detections
        #         })

        #         object_name = "unsure"
        #         object_score = 1
        #         if len(detections) > 0:
        #             object_name = detections[0][0]
        #             object_score = detections[0][1]

        #         if object_name not in products:
        #             products[object_name] = 0

        #         products[object_name] += 1

        #         if "coke" not in brands:
        #             brands["coke"] = 0

        #         if object_name != "unsure":
        #             brands["coke"] += 1
        #         else:
        #             brands["unsure"] += 1

        #         draw.rectangle(area, fill=None, outline=(0,255,0,120))
        #         draw.text([(int(box[1] * width), int(box[0] * height) - 20)],
        #             object_name, 
        #             fill=(0,255,0,120))
        #         draw.text([(int(box[1] * width), int(box[0] * height) - 10)],
        #             str(object_score), 
        #             fill=(0,255,0,120))

        #         # result.append({
        #         #     'box': box,
        #         #     'score': scores[i],
        #         #     'name': detections
        #         # })

        # org_image.save(folder_name + "/result.jpg", 'JPEG')
        # with open(folder_name + "/result.json", 'w') as outfile:
        #     json.dump({
        #         "products": products,
        #         "brands": brands
        #     }, outfile)

        # return jsonify(
            # products=products,
            # brands=brands,
            # job_id=folder_name
        # )


# @app.route('/upload/json', methods=['POST'])
# def upload_json():
#     file = request.files['image']
#     if file:
#         data = file.read()
#         org_image = Image.open(BytesIO(data))
#         for orientation in ExifTags.TAGS.keys() : 
#             if ExifTags.TAGS[orientation]=='Orientation' : break
#         if org_image._getexif() != None:
#             exif=dict(org_image._getexif().items())
#             if orientation in exif:
#                 if   exif[orientation] == 3 : 
#                     org_image=org_image.rotate(180, expand=True)
#                 elif exif[orientation] == 6 : 
#                     org_image=org_image.rotate(270, expand=True)
#                 elif exif[orientation] == 8 : 
#                     org_image=org_image.rotate(90, expand=True)

#         size = 1600,1600
#         org_image.thumbnail(size,Image.ANTIALIAS)
        
#         image = detect_objects(org_image)

#         draw = ImageDraw.Draw(org_image)

#         # select scores
#         result = []
#         folder_name = uuid.uuid4().hex
#         if os.path.isdir(folder_name) == False:
#             os.mkdir(folder_name)

#         boxes = image[0].tolist()
#         scores = image[1].tolist()
#         products = {}
#         brands = {}
#         brands["unsure"] = 0

#         for i in range(0, len(scores) - 1):
#             if scores[i] > 0.8:
#                 width, height = org_image.size
#                 box = boxes[i]
#                 area = [
#                     int(box[1] * width),
#                     int(box[0] * height),
#                     int(box[3] * width),
#                     int(box[2] * height)]
#                 # crop image
#                 out_path = folder_name + "/CROP_" + str(i) + ".JPG"
#                 org_image.crop(area).save(out_path, 'JPEG')
#                 cropped_img = tf.gfile.FastGFile(out_path, 'rb').read()
#                 detections = classify_object(cropped_img)

#                 result.append({
#                     'box': area,
#                     'score': scores[i],
#                     'name': detections
#                 })

#         return jsonify(result)

# @app.route('/upload/image', methods=['POST'])
# def upload_image():
#     file = request.files['image']
#     if file:
#         data = file.read()
#         org_image = Image.open(BytesIO(data))
#         image = detect_objects(org_image)

#         # select scores
#         result = []
#         draw = ImageDraw.Draw(org_image)


#         boxes = image[0].tolist()
#         scores = image[1].tolist()
#         for i in range(0, len(scores) - 1):
#             if scores[i] > 0.6:
#                 width, height = org_image.size
#                 box = boxes[i]
#                 area = [
#                     int(box[1] * width),
#                     int(box[0] * height),
#                     int(box[3] * width),
#                     int(box[2] * height)]
#                 # crop image
#                 out_path = "CROP_" + str(i) + ".JPG"
#                 org_image.crop(area).save(out_path, 'JPEG')
#                 cropped_img = tf.gfile.FastGFile(out_path, 'rb').read()
#                 detections = classify_object(cropped_img)

#                 object_name = "undefined"
#                 if len(detections) > 0:
#                     object_name = detections[0][0]
#                 draw.rectangle(area, fill=None, outline=(0,255,0,120))
#                 draw.text([(int(box[1] * width), int(box[0] * height) - 20)],
#                     object_name, 
#                     fill=(0,255,0,120))

#                 result.append({
#                     'box': box,
#                     'score': scores[i],
#                     'name': detections
#                 })


#         byte_io = BytesIO()
#         org_image.save(byte_io, 'JPEG')
#         byte_io.seek(0)

#         return send_file(byte_io, mimetype='image/jpeg')

# @app.route('/url/image')
# def url_image():
#     default_url = 'http://weather.goodmaps.co/IMG_1426.jpg'
#     url = request.args.get('url', default_url)
#     r = requests.get(url)
#     org_image = Image.open(BytesIO(r.content))
#     image = detect_objects(org_image)

#     # select scores
#     result = []
#     draw = ImageDraw.Draw(org_image)

#     boxes = image[0].tolist()
#     scores = image[1].tolist()
#     for i in range(0, len(scores) - 1):
#         if scores[i] > 0.6:
#             width, height = org_image.size
#             box = boxes[i]
#             area = [
#                 int(box[1] * width),
#                 int(box[0] * height),
#                 int(box[3] * width),
#                 int(box[2] * height)]
#             # crop image
#             out_path = "CROP_" + str(i) + ".JPG"
#             org_image.crop(area).save(out_path, 'JPEG')
#             cropped_img = tf.gfile.FastGFile(out_path, 'rb').read()
#             detections = classify_object(cropped_img)

#             draw.rectangle(area, fill=None, outline=(0,255,0,120))
#             draw.text([(int(box[1] * width), int(box[0] * height) - 20)],
#                 detections[0][0], 
#                 fill=(0,255,0,120))

#             result.append({
#                 'box': box,
#                 'score': scores[i],
#                 'name': detections
#             })


#     byte_io = BytesIO()
#     org_image.save(byte_io, 'JPEG')
#     byte_io.seek(0)

#     return send_file(byte_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)

