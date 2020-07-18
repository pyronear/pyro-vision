# -*- coding: utf-8 -*-

# Copyright (c) Pyronear contributors.
# This file is dual licensed under the terms of the CeCILL-2.1 and AGPLv3 licenses.
# See the LICENSE file in the root of this repository for complete details.

from picamera import PiCamera
from time import sleep
import glob
import torchvision
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import smtplib
import ssl


class PyronearEngine:
    """This class is the Pyronear Engine. This engine manage the whole Fire Detection
       process by capturing and saving the image and by predicting if there is a fire or
       not based on this image.
    Example
    -------
    pyronearEngine = PyronearEngine()
    pyronearEngine.run(30)  # For a prediction every 30s
    """
    def __init__(self, imgsFolder):
        # Camera
        self.camera = PiCamera()
        self.camera.rotation = 270

        # Images Folder
        self.imgsFolder = imgsFolder

        # Model definition
        self.model = torchvision.models.resnet18(pretrained=True)

        # Change fc
        in_features = getattr(self.model, 'fc').in_features
        setattr(self.model, 'fc', nn.Linear(in_features, 2))

        self.model.load_state_dict(torch.load("model/model_resnet18.txt", map_location=torch.device('cpu')))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.tf = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      normalize])

    def run(self, timeStep):

        imgs = glob.glob(self.imgsFolder + "//*.jpg")
        idx = len(imgs)

        while True:
            imagePath = self.imgsFolder + "//" + str(idx).zfill(8) + ".jpg"
            print(imagePath)
            self.capture(imagePath)
            pred = self.predict(imagePath)
            print(pred)
            idx = idx + 1
            sleep(timeStep)

    def capture(self, imagePath):

        self.camera.start_preview()
        sleep(3)  # gives the camera’s sensor time to sense the light levels
        self.camera.capture(imagePath)
        self.camera.stop_preview()

    def predict(self, imagePath):
        im = Image.open(imagePath)
        imT = self.tf(im)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(imT.unsqueeze(0))

        if pred[0, 0] > pred[0, 1]:
            return "no fire"
        else:
            sendAlert()
            return "FIRE !!!"


def sendAlert():
    port = 587  # For starttls
    smtp_server = "smtp.gmail.com"
    sender_email = "test.pyronear@gmail.com"  # Enter your address
    receiver_email = "mateo.lostanlen@gmail.com"  # Enter receiver address
    # password = ""  # uncomment and add your password
    message = """\
    Subject: FIRE

    Pyronear has detected a fire !!!"""

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port) as server:
        server.ehlo()  # Can be omitted
        server.starttls(context=context)
        server.ehlo()  # Can be omitted
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)


if __name__ == "__main__":

    pyronearEngine = PyronearEngine('DS')
    pyronearEngine.run(5)  # For a prediction every 5s
