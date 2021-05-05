import cv2
import numpy as np 
from os import remove, listdir
from os.path import isfile, join
import time
#load_yolo & detect_objects implemented using: https://github.com/nandinib1999/object-detection-yolo-opencv
def load_yolo():
	net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
	classes = []
	with open("coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]

	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers


def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.004, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def delete_person(outputs,img_path):
    for output in outputs:
        for detect in output:
            class_id = np.argmax(detect)
            conf = detect[class_id]
            if conf > 0.5 and class_id==0: #If a person in detected with over .5 confidence
                remove(img_path)
                return True
                
    return False

def image_detect(img_path): 
    model, classes, colors, output_layers = load_yolo()
    image = cv2.imread(img_path)
    blob, outputs = detect_objects(image, model, output_layers)
    delete = delete_person(outputs,img_path)
    if delete:
        print("Deleted!")
    else:
        print("Still here")

mypath = "test"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
start_time = time.time()
for x in onlyfiles:
    image_detect(mypath+"\\"+x)
print("--- %s seconds ---" % (time.time() - start_time))
