import unittest
import torch
import numpy as np
import random
from pyronear import models


def set_rng_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_available_classification_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


class ModelsTester(unittest.TestCase):

    def _test_classification_model(self, name, input_shape):
        # passing num_class equal to a number other than default helps in making the test
        # more enforcing in nature
        set_rng_seed(0)
        num_classes = 50

        # Pretrained parameters
        self.assertRaises(ValueError, models.__dict__[name], pretrained=True, imagenet_pretrained=True)

        # Default case
        model = models.__dict__[name](num_classes=num_classes)
        model.eval()
        x = torch.rand(input_shape)
        with torch.no_grad():
            out = model(x)
        # self.assertExpected(out, rtol=1e-2, atol=0.)
        self.assertEqual(out.shape[-1], 50)


for model_name in get_available_classification_models():
    # for-loop bodies don't define scopes, so we have to save the variables
    # we want to close over in some way
    def do_test(self, model_name=model_name):
        input_shape = (1, 3, 224, 224)
        self._test_classification_model(model_name, input_shape)

    setattr(ModelsTester, "test_" + model_name, do_test)


if __name__ == '__main__':
    unittest.main()
