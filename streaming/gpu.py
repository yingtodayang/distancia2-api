
from streaming.yoloTF2 import *


class Network:
    def __init__(self, weightsPath, configPath, labelsPath, **kwargs):
        self.model_size = (416, 416, 3)
        self.gpu_name = kwargs['gpu_name']
        self.max_output_size = 40
        self.max_output_size_per_class= 20
        self.iou_threshold = kwargs['threshold']
        self.confidence_threshold = kwargs['confidence']
        labels = open(labelsPath).read().strip().split("\n")
        with tf.device(self.gpu_name):
            self.net = YOLOv4Net(configPath, labelsPath)
            self.net.load_weights(weightsPath,weights_type='yolo')
            self.net.summary(summary_type="yolo")
            self.net.summary()

    def make_boxes(self, image):
        # image = np.array(image)
        # image = tf.expand_dims(image, 0)
        resized_frame = self.net.resize_image(image)
        image = np.array(image)

        with tf.device(self.gpu_name):
            pred = self.net.predict(image, prob_thresh=self.confidence_threshold)
            # boxes, scores, classes, nums = output_boxes( \
            # pred, self.model_size,
            # max_output_size=self.max_output_size,
            # max_output_size_per_class=self.max_output_size_per_class,
            # iou_threshold=self.iou_threshold,
            # confidence_threshold=self.confidence_threshold)

        # boxes, objectness, classes, nums = boxes[0], scores[0], classes[0], nums[0]
        # boxes=np.array(boxes)

        Fboxes = []
        Fconfidences = []
        FclassIDs = []  
        #print(pred[0][5])
        for box in pred:
            if int(box[4] == 0): # SOLO DETECTAMOS PERSONAS.
                # img = np.squeeze(image)
                # x1y1 = tuple((boxes[i,0:2] * [img.shape[1],img.shape[0]]).astype(np.int32))
                # x2y2 = tuple((boxes[i,2:4] * [img.shape[1],img.shape[0]]).astype(np.int32))
                # a = [0,0,0,0]
                # a[0] = x1y1[0]
                # a[1] = x1y1[1]
                # a[2] = x2y2[0] - x1y1[0]
                # a[3] = x2y2[1] - x1y1[1]
                # a = boxes[i]
                Fboxes.append(box)
        #         Fconfidences.append(float(objectness[i]))
        #         FclassIDs.append(int(classes[i]))

        return Fboxes

    def draw_image(self,image,boxes):
        return self.net.draw_bboxes(image,boxes)

