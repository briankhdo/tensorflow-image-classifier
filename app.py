#!/usr/bin/python3
import re
from io import BytesIO
import decimal
import flask.json
from flask import Flask, send_file, request, jsonify, render_template, send_from_directory
from PIL import Image, ImageDraw, ExifTags, ImageFont
import requests
import numpy as np
import tensorflow as tf
import tensorflow as tf_classify
import os
import json
import uuid
import base64
import copy

import cv2

from align_dlib import AlignDlib

config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 1

num_of_photos = {'user_checkin_anh.ph': 61,
'user_checkin_anhdd': 36,
'user_checkin_anhhnv': 37,
'user_checkin_anhhtt': 32,
'user_checkin_anhld': 100,
'user_checkin_anhlht': 16,
'user_checkin_anhltk': 4,
'user_checkin_anhlv': 4,
'user_checkin_anhnt': 83,
'user_checkin_anhnt92': 5,
'user_checkin_anhntq': 22,
'user_checkin_anhpq': 55,
'user_checkin_anhtb': 7,
'user_checkin_anhttn': 17,
'user_checkin_anhvd': 19,
'user_checkin_anhvt': 34,
'user_checkin_bachv': 8,
'user_checkin_balh': 15,
'user_checkin_bangnh': 50,
'user_checkin_ceo': 7,
'user_checkin_chienpd': 38,
'user_checkin_chinhtv': 23,
'user_checkin_chivtp': 69,
'user_checkin_chungnt': 23,
'user_checkin_congpm': 4,
'user_checkin_cudc': 91,
'user_checkin_cuongnx': 5,
'user_checkin_dienlt': 16,
'user_checkin_diepdh': 49,
'user_checkin_dinhtd': 16,
'user_checkin_dongnt': 49,
'user_checkin_ducnt1': 68,
'user_checkin_ducnt93': 58,
'user_checkin_ducnv': 72,
'user_checkin_dungdt': 12,
'user_checkin_dunght': 36,
'user_checkin_dungna': 36,
'user_checkin_dungnh': 56,
'user_checkin_dungnt1': 35,
'user_checkin_dungnt90': 7,
'user_checkin_dungpt': 41,
'user_checkin_dungtv': 44,
'user_checkin_duongbv': 36,
'user_checkin_duongdt': 21,
'user_checkin_duongnt1': 26,
'user_checkin_duongvt': 27,
'user_checkin_duynk': 29,
'user_checkin_duynt': 32,
'user_checkin_duypb': 40,
'user_checkin_duyvd': 42,
'user_checkin_giangnh': 45,
'user_checkin_giangnth': 21,
'user_checkin_haidt': 29,
'user_checkin_hailt': 26,
'user_checkin_haipt': 45,
'user_checkin_hangbt': 53,
'user_checkin_hangbt95': 34,
'user_checkin_hangct': 13,
'user_checkin_hangntd': 57,
'user_checkin_hangntt': 52,
'user_checkin_hanh': 9,
'user_checkin_hanv': 10,
'user_checkin_haontp': 20,
'user_checkin_haptt': 27,
'user_checkin_haunt': 8,
'user_checkin_haunth': 11,
'user_checkin_hienntt': 21,
'user_checkin_hieplt': 68,
'user_checkin_hiepvs': 16,
'user_checkin_hieuht': 16,
'user_checkin_hieult': 29,
'user_checkin_hieunt86': 4,
'user_checkin_hieupc': 92,
'user_checkin_hoanglm': 18,
'user_checkin_hoangnv': 5,
'user_checkin_hoanna': 47,
'user_checkin_hoavv': 32,
'user_checkin_hopnv': 49,
'user_checkin_hunghv': 42,
'user_checkin_hungnk': 30,
'user_checkin_hungscv': 57,
'user_checkin_huongdt1': 15,
'user_checkin_huongdtx': 15,
'user_checkin_huongnt': 39,
'user_checkin_huongnt94': 32,
'user_checkin_huongpt1': 38,
'user_checkin_huongtt': 29,
'user_checkin_huutq': 58,
'user_checkin_huyenht': 10,
'user_checkin_huyenlt90': 12,
'user_checkin_huyennd': 80,
'user_checkin_huyennt': 33,
'user_checkin_huyenntt': 37,
'user_checkin_huyenpt': 11,
'user_checkin_huyentt94': 10,
'user_checkin_huynd': 19,
'user_checkin_huynd91': 27,
'user_checkin_huynm': 5,
'user_checkin_huynq': 15,
'user_checkin_khaivq': 30,
'user_checkin_khangnn': 10,
'user_checkin_khanhdd': 14,
'user_checkin_khanhnq1': 5,
'user_checkin_khanhpt': 19,
'user_checkin_khanhqd': 40,
'user_checkin_khoatv': 53,
'user_checkin_khuedb': 9,
'user_checkin_kiennt': 50,
'user_checkin_lannm': 58,
'user_checkin_lienptb': 22,
'user_checkin_linhcm': 23,
'user_checkin_linhdt': 15,
'user_checkin_linhdtm': 25,
'user_checkin_linhlb': 18,
'user_checkin_linhlm': 13,
'user_checkin_linhln': 28,
'user_checkin_linhng': 7,
'user_checkin_linhnh': 17,
'user_checkin_linhntk': 46,
'user_checkin_linhtdt': 10,
'user_checkin_linhtn': 16,
'user_checkin_linhvd': 7,
'user_checkin_loannt': 20,
'user_checkin_locnt': 22,
'user_checkin_longdt': 14,
'user_checkin_longhp': 18,
'user_checkin_longht': 57,
'user_checkin_longnd': 85,
'user_checkin_longvt': 8,
'user_checkin_luannb': 35,
'user_checkin_lynh': 62,
'user_checkin_lyntk': 4,
'user_checkin_maidtt': 8,
'user_checkin_maintt': 5,
'user_checkin_manhnv': 37,
'user_checkin_minhlb1': 5,
'user_checkin_minhln': 65,
'user_checkin_muonnt': 11,
'user_checkin_mynt': 19,
'user_checkin_namhh': 21,
'user_checkin_nampv': 4,
'user_checkin_ngadt1': 4,
'user_checkin_ngant': 4,
'user_checkin_ngapq': 37,
'user_checkin_ngocld': 5,
'user_checkin_ngocndb': 10,
'user_checkin_ngocnm': 31,
'user_checkin_ngocph': 18,
'user_checkin_nguyennt': 42,
'user_checkin_nhannt': 19,
'user_checkin_nhatla': 14,
'user_checkin_nhudtx': 4,
'user_checkin_nhungnth': 46,
'user_checkin_nhungtt': 22,
'user_checkin_nhungvtt': 20,
'user_checkin_oanhntd': 37,
'user_checkin_phonglq': 4,
'user_checkin_phongth': 15,
'user_checkin_phucnh': 9,
'user_checkin_phuongdm90': 55,
'user_checkin_phuongnm': 83,
'user_checkin_phuongnt': 25,
'user_checkin_phutc': 28,
'user_checkin_quandh': 29,
'user_checkin_quanglv': 16,
'user_checkin_quangpv': 112,
'user_checkin_quangta': 51,
'user_checkin_quanna': 11,
'user_checkin_quannh': 50,
'user_checkin_quannm': 28,
'user_checkin_quyetpv': 36,
'user_checkin_quynhnt': 40,
'user_checkin_runaway518': 17,
'user_checkin_soncb': 5,
'user_checkin_sondn': 22,
'user_checkin_sondt': 39,
'user_checkin_sonnt': 8,
'user_checkin_tampv': 40,
'user_checkin_tanmn': 59,
'user_checkin_tannt': 8,
'user_checkin_thangld': 4,
'user_checkin_thangnh': 40,
'user_checkin_thangpm': 10,
'user_checkin_thanhna': 27,
'user_checkin_thanhnb': 31,
'user_checkin_thanhnt': 36,
'user_checkin_thanhnth': 12,
'user_checkin_thaoltp': 11,
'user_checkin_thaont': 10,
'user_checkin_thaontp1': 30,
'user_checkin_thaopp': 28,
'user_checkin_thiendt': 59,
'user_checkin_thieubd': 42,
'user_checkin_thinhhv': 7,
'user_checkin_thinhvd': 13,
'user_checkin_thoantk': 9,
'user_checkin_thunth': 26,
'user_checkin_thuph': 49,
'user_checkin_thuttp': 5,
'user_checkin_thuych': 37,
'user_checkin_thuydt': 12,
'user_checkin_thuyld': 37,
'user_checkin_tientm': 22,
'user_checkin_tientm1': 6,
'user_checkin_tinld': 4,
'user_checkin_toannh': 11,
'user_checkin_trangdt': 26,
'user_checkin_tranglth': 17,
'user_checkin_trangnd': 28,
'user_checkin_trangnt': 106,
'user_checkin_trangnt2': 20,
'user_checkin_trangnth1': 14,
'user_checkin_trangtq': 25,
'user_checkin_trangvh': 15,
'user_checkin_trangvth': 32,
'user_checkin_trieuhv': 30,
'user_checkin_trinhdc': 92,
'user_checkin_tritdm': 5,
'user_checkin_trungpd': 47,
'user_checkin_truyennv': 30,
'user_checkin_tuanlm': 43,
'user_checkin_tuannm89': 41,
'user_checkin_tuannq': 44,
'user_checkin_tuannt': 41,
'user_checkin_tuanpm1': 53,
'user_checkin_tuantm': 10,
'user_checkin_tuent': 31,
'user_checkin_tungds': 52,
'user_checkin_tunglt': 67,
'user_checkin_tungnd': 37,
'user_checkin_tungnt': 37,
'user_checkin_tunm': 75,
'user_checkin_vanhg': 4,
'user_checkin_vidh': 11,
'user_checkin_viet751993': 66,
'user_checkin_vinhts': 29,
'user_checkin_vutd': 69,
'user_checkin_xuanda': 70,
'user_checkin_yendth': 24,
'user_checkin_yenlh': 6,
'user_checkin_yenlt': 63,
'user_checkin_yennh': 5,
'user_checkin_yennth': 36,
'user_checkin_yenth': 25,
'user_checkin_yentt': 4,
'user_checkin_yentt89': 27}

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
        with tf.gfile.GFile(PATH_TO_IMAGE_CKPT, 'rb') as fid:
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

label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile(PATH_TO_IMAGE_LABELS)]


def classify_faces(faces, prediction_sess):
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

@app.route('/images/<path:path>')
def send_js(path):
    return send_from_directory('images', path)


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

        images = []

        detections = []

        for index, model_name in enumerate(CKPT_MODEL_NAME):

            classify_results = classify_faces(faces)
            detections.append(classify_results)

            image = copy.copy(org_image)

            draw = ImageDraw.Draw(image)

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

                    posibility = 1
                    if len(face_classes) > 0:
                        possible_class = face_classes[0]
                        class_name = possible_class[0]
                        num_of_photo = num_of_photos['user_checkin_' + class_name.replace(' ', '_')]
                        face_classes[0].append(num_of_photo)
                        posibility = possible_class[1]

                    draw.rectangle(area, fill=None, outline=(0,255,0,120))
                    left = box.left()
                    if left < 0:
                        left = 0
                    draw.text([(left, box.top() - 30)],
                        class_name, 
                        fill=(0,255,0,120))
                    draw.text([(left, box.top() - 20)],
                        str(posibility), 
                        fill=(0,255,0,120))
                    draw.text([(left, box.top() - 10)],
                        str(num_of_photo),
                        fill=(0,255,0,120))

            else:
                draw.text([10, 10],
                    'Cannot detect faces', 
                    fill=(0,255,0,120))

            byte_io = BytesIO()
            file_name = "./images/%s_result.jpg" % uuid.uuid4().hex
            image.save(file_name, 'JPEG')
            images.append(file_name)

        return render_template('upload.html', models=CKPT_MODEL_NAME, images=images, detections=detections)
        # return send_file(byte_io, mimetype='image/jpeg')


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

