import torch

from sam2.build_sam import build_sam2


def test_build_sam(download_weights):
    model = build_sam2(
        "sam2_hiera_t.yaml", "./artifacts/sam2_hiera_tiny.pt", device="cpu"
    )

    assert isinstance(model, torch.nn.Module)
