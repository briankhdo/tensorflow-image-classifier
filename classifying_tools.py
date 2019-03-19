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

