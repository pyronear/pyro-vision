# -*- coding: utf-8 -*-

# Copyright (c) Pyronear contributors.
# This file is dual licensed under the terms of the CeCILL-2.1 and AGPLv3 licenses.
# See the LICENSE file in the root of this repository for complete details.


import torchvision
import torch
from torch import nn
from PIL import Image
from torchvision import transforms


class PyronearPredictor:
    """This class use the last pyronear model to predict if a sigle frame contain a fire or not 

    Example
    -------
    pyronearPredictor = PyronearPredictor()

    res = pyronearPredictor.predict("test.jpg")
    print(res)

    """
    def __init__(self):
        # Model definition
		self.model = torchvision.models.resnet18(pretrained=True)

		# Change fc
		in_features = getattr(self.model, 'fc').in_features
		setattr(self.model, 'fc', nn.Linear(in_features, 2))

		self.model.load_state_dict(torch.load("model/model_resnet18.txt"))

		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

		self.tf = transforms.Compose([
		    transforms.Resize((224,224)),
		    transforms.ToTensor(),
		    normalize
		])

    def predict(self, imPath):
        im = Image.open(imPath)
		imT = self.tf(im)

		model.eval()
		with torch.no_grad():
		    pred = self.model(imT.unsqueeze(0))
		    
		if pred[0,0]>pred[0,1]:
		    return "no fire"
		else:
		    return "FIRE !!!"
