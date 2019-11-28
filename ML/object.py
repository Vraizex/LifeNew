from imageai.Detection import ObjectDetection

detector = ObjectDetection()


# model_path = 'C:/Users/User/Downloads/segnet_best.h5'
input_path = "C:/Users/User/Downloads/q1.jpg"
model_path = 'C:/Users/User/Downloads/yolo-tiny.h5'
output_path = 'C:/Users/User/Downloads/newimage.jpg'

detector.setModelTypeAsTinyYOLOv3()

# detector.setModelTypeAsRetinaNet()

detector.setModelPath(model_path)
detector.loadModel()
detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)
for eachItem in detection:
    print(eachItem["name"], " : ", eachItem["percentage_probability"])
    print("--------------------------------")
# model_path = 'C:/Users/User/Downloads/NetWork.h5'
# model_path = 'C:/Users/User/Downloads/yolo-tiny.h5'
# output_path = 'C:/Users/User/Downloads/newimage.jpg'
# detector.setModelTypeAsRetinaNet()

# detector.setModelTypeAsYOLOv3()
# detector.setModelTypeAsTinyYOLOv3()
# detector.setModelPath(model_path)
# detector.loadModel()

# detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path,
#                                           minimum_percentage_probability=30)
# for eachItem in detection:
#    print(eachItem["name"], " : ", eachItem["percentage_probability"])
#    print("--------------------------------")
