from imageai.Detection import ObjectDetection
import os
from Bard import Chatbot

Secure_1PSID = 'eQiF10azK3PM_zjdEJ4IAyDGRQQaR1Y4iqM8zfRk_ENerBQ_spFB01tqzbVrgyBmsZd6HA.'
Secure_1PSIDTS = 'sidts-CjIBPVxjSg-5e_NCuu3i4RhQUQcAyjIOU6iq_J0GXPS04Wi6Bbg2iSFOBWZPQmTn-g7CQhAA'

execution_path = os.getcwd()
print(execution_path)

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("/Users/joe/Documents/GitHub/yolov3.pt")
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))

labels = ""

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")
    labels = labels + eachObject["name"] + ", "

chatbot = Chatbot(Secure_1PSID, Secure_1PSIDTS)
query = "You are a visual storyteller. Tell a 300 word story with the objects: " + labels
answer = chatbot.ask(query)
print(answer['content'])