
import os
import logging
import itertools

from django.conf import settings
from scipy.spatial import distance
import numpy as np
import cv2

if settings.MODEL_ENABLE_GPU:
    from streaming.gpu import Network
else:
    from streaming.cpu import Network

logger = logging.getLogger('ProcesingRoutine')

ORI_X = 0
ORI_Y = 1
WIDTH = 2
HEIGHT = 3

PERSON_CLASS_ID = 0
COLOR_PEOPLE_BOX = (99,  43, 206)
COLOR_FAR_LINE = (102, 220, 225)
COLOR_CLOSE_LINE = (36,60, 255)
LIGHT_COLOR_TEXT = (255,255,255)
DARK_COLOR_TEXT = (0, 0, 0)
THICKNESS_LINE = 2

class CamProcessor:
    def __init__(self, processor_name, **kwargs):
        self.people_height = settings.MODEL_PEOPLE_HEIGHT

        network_settings = {
            'weightsPath': settings.BASE_DIR(settings.MODEL_WEIGHTS_PATH_V3) if settings.MODEL_YOLO == 'yolov3' else settings.BASE_DIR(settings.MODEL_WEIGHTS_PATH_V4),
            'configPath': settings.BASE_DIR(settings.MODEL_CONFIG_PATH_V3) if settings.MODEL_YOLO == 'yolov3' else settings.BASE_DIR(settings.MODEL_CONFIG_PATH_V4),
            'labelsPath': settings.BASE_DIR(settings.MODEL_LABELS_PATH),
            'threshold': settings.MODEL_THRESHOLD,
            'confidence': settings.MODEL_CONFIDENCE,
        }

        if settings.MODEL_ENABLE_GPU:
            network_settings['gpu_name'] = processor_name
            logger.info('USANDO GPU')
        try:
            self.net = Network(**network_settings)
        except Exception as error:
            logger.error(error)

    def distance_measure(self, boxes, image):
        if(settings.MODEL_ENABLE_GPU and settings.MODEL_YOLO == 'yolov4'):
            get_lower_center = lambda box: ((box[ORI_X] + (box[WIDTH] / 4)), box[ORI_Y]+(box[HEIGHT]/2))
        else:
            get_lower_center = lambda box: (box[ORI_X] + box[WIDTH] // 2, box[ORI_Y] + box[HEIGHT])
        
        f = lambda x: np.arctan(x) / (np.pi/2)

        if(len(boxes) > 0 and settings.MODEL_ENABLE_GPU and settings.MODEL_YOLO=='yolov4'):
            height, width, _ = image.shape
            boxes = boxes * np.array([width, height, width, height, 1, 1])

        results = []
        
        for (index_a,box_a), (index_b, box_b) in itertools.combinations(enumerate(boxes), 2):
            base_box_a = get_lower_center(box_a)
            base_box_b = get_lower_center(box_b)

            euclidean_distance = distance.euclidean(base_box_a, base_box_b)
            height_box_a = self.people_height / box_a[HEIGHT]
            height_box_b = self.people_height / box_b[HEIGHT]

            l1 = f(box_a[HEIGHT] / box_b[HEIGHT])
            l2 = 1 - l1

            D = l1 * height_box_a * euclidean_distance + l2 * height_box_b * euclidean_distance

            a = [index_a, index_b,
                 base_box_a, base_box_b,
                 round(D, 2)]
            results.append(a)

        return np.array(results,dtype=object)
        

    def get_min_distances(self, distances):
        min_distances = []
        if len(distances)>0:
            for i in set(distances[:,0]):
                a = distances[distances[:,0] == i][distances[distances[:,0] == i][:,-1].argmin()]
                min_distances.append(a)

        return np.array(min_distances)

    def draw_over_frame(self, image, boxes, distance_lines):
        if settings.MODEL_ENABLE_GPU and settings.MODEL_YOLO == 'yolov4':
            image = self.net.draw_image(image,boxes)
        else:
            for box in boxes:
                edge_0 = (box[ORI_X], box[ORI_Y])
                edge_1 = (box[ORI_X] + box[WIDTH], box[ORI_Y] + box[HEIGHT])
                cv2.rectangle(image, edge_0, edge_1, COLOR_PEOPLE_BOX, THICKNESS_LINE)
        
        for line in distance_lines:
            line_color = COLOR_CLOSE_LINE if line[4] < settings.SECURE_DISTANCE else COLOR_FAR_LINE
            if(settings.MODEL_ENABLE_GPU and settings.MODEL_YOLO == 'yolov4'):
                a = np.float32(line[2][0])
                b = np.float32(line[2][1])
                c = np.float32(line[3][0])
                d = np.float32(line[3][1])
                cv2.line(image, (a,b), (c,d), line_color, THICKNESS_LINE)
            else:
                image = cv2.line(image, line[2], line[3], line_color, THICKNESS_LINE)
            
            e = ((np.array(line[2])+np.array(line[3]))/2).astype(int)
            mesure_text = '{}m'.format(line[4])
            position_text = (e[0], e[1] - 5)
            cv2.putText(image, mesure_text, position_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, DARK_COLOR_TEXT, THICKNESS_LINE)
            cv2.putText(image, mesure_text, position_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, LIGHT_COLOR_TEXT, THICKNESS_LINE - 1)

        return image

    def calculate_statistical_results(self, boxes, distance_lines):
        amount_people = len(boxes)

        if amount_people > 1:
            minimal_distance = min([line[4] for line in distance_lines])
            breaking_secure_distance = sum([line[4] < settings.SECURE_DISTANCE for line in distance_lines])
            average_distance = np.mean([line[4] for line in distance_lines])
        else:
            minimal_distance = 0
            breaking_secure_distance = 0
            average_distance = 0
        
        return {
            'amount_people': amount_people,
            'breaking_secure_distance': breaking_secure_distance,
            'minimal_distance': minimal_distance,
            'average_distance': average_distance,
        }

    def inference(self, frame):
        frame = cv2.resize(frame, (800, 600),
            fx=0, fy=0, interpolation=cv2.INTER_AREA)
        boxes = self.net.make_boxes(frame)
        distance_lines = self.distance_measure(boxes, frame)
        shorter_distance_lines = self.get_min_distances(distance_lines)
        statistical = self.calculate_statistical_results(boxes, shorter_distance_lines)
        if statistical['amount_people'] > 0:
            frame = self.draw_over_frame(frame, boxes, shorter_distance_lines)
        
        results = {
            'frame': frame,
            'graphical': {
                'boxes': boxes,
                'distance_lines': distance_lines,
                'shorter_distance_lines': shorter_distance_lines
            },
            'statistical': statistical
        }

        return results
