from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2
import rospy
import math

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        #self.MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
        self.MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
        model_file = self.MODEL_NAME + '/frozen_inference_graph.pb'

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph)
            # Input and output Tensors for detection_graph
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def get_average_color(self, rgb_img):
        """Get the average value for each r g b channel in the cropped rgb image
        """
        val = [0,0,0]
        for i in range(3):
            img = rgb_img[:,:,i]
            avg_color_per_row = np.average(img, axis=0)
            val[i] = np.average(avg_color_per_row, axis=0)
            
        return val
    
    def determine_color(self, val):
        """Determines the closest color 
        red is 255, 0, 0
        yellow is 255, 255, 0
        green is 0, 255, 0
        """
        colors = [[255, 0, 0],[255, 255, 0],[0, 255, 0]]
        diff = float("inf")
        diff_index = -1
        for i in range(len(colors)):
            current_diff = (val[0]-colors[i][0])**2 + (val[1]-colors[i][1])**2 + (val[2]-colors[i][2])**2
            if current_diff < diff:
                diff = current_diff
                diff_index = i
                
        if diff_index == 0:
            return TrafficLight.RED
        elif diff_index == 1:
            return TrafficLight.YELLOW
        elif diff_index == 2:
            return TrafficLight.GREEN
        else:
            return TrafficLight.UNKNOWN
            
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = np.asarray(image, dtype="uint8")
        image_np_expanded = np.expand_dims(image_np, axis=0)
        
        detected = False

        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        best_scores = []
        
        for idx, classID in enumerate(classes):
            if self.MODEL_NAME == 'ssd_mobilenet_v1_coco_2017_11_17':
                if classID == 10: # 10 is traffic light
                    if scores[idx] > 0.15: #confidence level
                        best_scores.append([scores[idx], idx, classID])
                        detected = True
        
        if detected:
            # Sort to get the best score
            best_scores.sort(key=lambda tup: tup[0], reverse=True)
            best_score = best_scores[0]
            nbox = boxes[best_score[1]]
            # Get bounding box for this object
            height = image.shape[0]
            width = image.shape[1]
            box = np.array([nbox[0]*height, nbox[1]*width, nbox[2]*height, nbox[3]*width]).astype(int)
            cropped_image = image[box[0]:box[2], box[1]:box[3]]
            # Get average color 
            val = self.get_average_color(cropped_image)
            # Predict color
            traffic_light = self.determine_color(val)
            # Draw bounding box with detected color
            traffic_light_color = (255,255,255)#black 
            if traffic_light==TrafficLight.RED:
                traffic_light_color = (255,0,0)
            elif traffic_light == TrafficLight.YELLOW:
                traffic_light_color = (255,255,0)
            elif traffic_light == TrafficLight.GREEN:
                traffic_light_color = (0,255,0)
            cv2.rectangle(image, (box[1],box[0]), (box[3],box[2]),traffic_light_color,4)
            return traffic_light
            
        return TrafficLight.UNKNOWN
