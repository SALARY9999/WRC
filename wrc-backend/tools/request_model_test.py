"""Send sample.json plus its sixteen image paths to the offline API."""
import argparse
import base64
import json
from pathlib import Path
import sys
import urllib.request

# Run as ``python tools/request_model_test.py``, sys.path[0] is ``tools/``, so
# the backend root has to be named before ``config.loader`` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.loader import load_network_config  # noqa: E402


def parse_args(argv=None):
    """The URL default is resolved per call, not at import, so a test can move
    the document and see the client follow it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path)
    parser.add_argument(
        "--url",
        default=load_network_config().package_test_url,
        help="offline test API URL (default: network.http_client.package_test_url)",
    )
    parser.add_argument("--output", type=Path, default=Path("model_result.json"))
    return parser.parse_args(argv)


def main():
    args = parse_args()
    payload = json.loads(args.sample.read_text(encoding="utf-8"))
    for camera in payload["cameras"]:
        camera["images_base64"] = [base64.b64encode(
            (args.sample.parent / path).read_bytes()).decode("ascii")
            for path in camera.pop("image_paths")]
    request = urllib.request.Request(args.url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
