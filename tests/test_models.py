import pytest
import torch

from pyrovision import models


def _test_classification_model(name, num_classes=10):

    batch_size = 2
    x = torch.rand((batch_size, 3, 224, 224))
    model = models.__dict__[name](pretrained=True).eval()
    with torch.no_grad():
        out = model(x)

    assert out.shape[0] == x.shape[0]
    assert out.shape[-1] == num_classes

    # Check backprop is OK
    target = torch.zeros(batch_size, dtype=torch.long)
    model.train()
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, target)
    loss.backward()


@pytest.mark.parametrize(
    "arch",
    [
        'mobilenet_v3_small', 'mobilenet_v3_large',
        'resnet18', 'resnet34',
        'rexnet1_0x', 'rexnet1_3x', 'rexnet1_5x',
    ],
)
def test_classification_model(arch):
    _test_classification_model(arch, 10 if arch.startswith("rexnet_") else 1000)
