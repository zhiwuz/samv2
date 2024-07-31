import pytest
import torch

from sam2.build_sam import build_sam2


@pytest.mark.full
@pytest.mark.parametrize(
    "variant",
    ["tiny", "small", "base_plus", "large"],
)
def test_build_sam(download_weights, variant: str):
    model = build_sam2(
        "sam2_hiera_t.yaml", "./artifacts/sam2_hiera_tiny.pt", device="cpu"
    )

    assert isinstance(model, torch.nn.Module)
