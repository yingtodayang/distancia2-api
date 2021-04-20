import logging
import tensorflow as tf
from yolov4.tf import YOLOv4
import numpy as np
from scipy.spatial import distance
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, ZeroPadding2D, LeakyReLU, UpSampling2D
from django.conf import settings
import cv2

logger = logging.getLogger()

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(device, True)
        logger.info('USING GPU')
    except:
        logger.error(f'Invalid device or cannot modify virtual devices once initialized {device}')

# def parse_cfg(cfgfile):
#     with open(cfgfile, 'r') as file:
#         lines = [line.rstrip('\n') for line in file if line != '\n' and line[0] != '#']
#     holder = {}
#     blocks = []
#     for line in lines:
#         if line[0] == '[':
#             line = 'type=' + line[1:-1].rstrip()
#             if len(holder) != 0:
#                 blocks.append(holder)
#                 holder = {}
#         key, value = line.split("=")
#         holder[key.rstrip()] = value.lstrip()
#     blocks.append(holder)
#     return blocks

def YOLOv4Net(configPath, labelsPath):
    yolo = YOLOv4()
    yolo.config.parse_names(labelsPath)
    yolo.config.parse_cfg(configPath)
    yolo.make_model()
    return yolo

# Esta funcion escala la imagen.
def resize_image(inputs, modelsize):
    inputs= tf.image.resize(inputs, modelsize)
    return inputs

def output_boxes(inputs,model_size, max_output_size, max_output_size_per_class, 
                 iou_threshold, confidence_threshold):

    center_x, center_y, width, height, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

    top_left_x = center_x - width / 2.0
    top_left_y = center_y - height / 2.0
    bottom_right_x = center_x + width / 2.0
    bottom_right_y = center_y + height / 2.0

    inputs = tf.concat([top_left_x, top_left_y, bottom_right_x,
                        bottom_right_y, confidence, classes], axis=-1)

    boxes_dicts = non_max_suppression(inputs, model_size, max_output_size, 
                                      max_output_size_per_class, iou_threshold, confidence_threshold)

    return boxes_dicts

def non_max_suppression(inputs, model_size, max_output_size, 
                        max_output_size_per_class, iou_threshold, confidence_threshold):
    bbox, confs, class_probs = tf.split(inputs, [4, 1, -1], axis=-1)
    bbox=bbox/model_size[0]
    scores = confs * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=max_output_size_per_class,
        max_total_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=confidence_threshold
    )
    return boxes, scores, classes, valid_detections
