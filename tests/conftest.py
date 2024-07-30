import subprocess

import pytest


@pytest.fixture
def download_weights() -> None:

    url: str = (
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
    )
    output_directory: str = "artifacts"

    command = ["wget", url, "-P", output_directory]

    try:
        # Execute the command
        result = subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print("Download completed successfully.")
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print("An error occurred during the download.")
        print(e.stderr.decode())
