import pytest
import torch

from pyrovision.models import utils


def _test_classification_hub_model(hub_repo, num_classes=1):

    batch_size = 2
    x = torch.rand((batch_size, 3, 224, 224))
    model = utils.model_from_hf_hub(hub_repo).eval()
    with torch.no_grad():
        out = model(x)

    assert out.shape[0] == x.shape[0]
    assert out.shape[-1] == num_classes

    # Check backprop is OK
    target = torch.zeros(batch_size, dtype=torch.long)
    model.train()
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, target)
    loss.backward()


@pytest.mark.parametrize(
    "hub_repo",
    [
        "pyronear/rexnet1_0x",
    ],
)
def test_model_from_hf_hub(hub_repo):
    _test_classification_hub_model(hub_repo, 1)
