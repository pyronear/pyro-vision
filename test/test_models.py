import unittest
import torch
from pyronear import models


class ModelsTester(unittest.TestCase):
    def test_resnet(self):

        # Test parameters
        img_shape = (3, 224, 224)
        bin_classif = True

        # Generated inputs
        img_tensor = torch.rand(img_shape)
        num_classes = 1 if bin_classif else None

        # Non-binary classification
        self.assertRaises(NotImplementedError, models.resnet,
                          depth=18, pretrained=True, bin_classif=False)

        # Working case
        model = models.resnet(depth=18, pretrained=True, bin_classif=True).eval()
        with torch.no_grad():
            out = model(img_tensor.unsqueeze(0))

        self.assertEqual(out.shape, (1, num_classes))


if __name__ == '__main__':
    unittest.main()
