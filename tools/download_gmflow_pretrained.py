#!/usr/bin/env python3
"""Download and extract the official single-scale GMFlow Things checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import shutil
import tempfile
from typing import Dict, Optional
import zipfile

import requests
import torch


OFFICIAL_FILE_ID = "1d5C5cgHIxWGsFR1vYs5XrQbbUiZl9TX2"
TARGET_NAME = "gmflow_things-e9887eda.pth"
OFFICIAL_SHARE_URL = (
    "https://drive.google.com/file/d/"
    f"{OFFICIAL_FILE_ID}/view?usp=sharing"
)


class _DownloadFormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.action: Optional[str] = None
        self.inputs: Dict[str, str] = {}
        self._inside_download_form = False

    def handle_starttag(self, tag: str, attrs) -> None:
        values = dict(attrs)
        if tag == "form" and (
            values.get("id") == "download-form"
            or "download" in values.get("action", "")
        ):
            self._inside_download_form = True
            self.action = values.get("action")
        elif tag == "input" and self._inside_download_form:
            name = values.get("name")
            if name is not None:
                self.inputs[name] = values.get("value", "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._inside_download_form:
            self._inside_download_form = False


def _binary_response(
    session: requests.Session,
    file_id: str,
) -> requests.Response:
    url = "https://drive.google.com/uc"
    response = session.get(
        url,
        params={"export": "download", "id": file_id},
        stream=True,
        timeout=(30, 300),
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if "text/html" not in content_type:
        return response

    html = response.text
    token = next(
        (
            value
            for key, value in response.cookies.items()
            if key.startswith("download_warning")
        ),
        None,
    )
    if token:
        confirmed = session.get(
            url,
            params={
                "export": "download",
                "id": file_id,
                "confirm": token,
            },
            stream=True,
            timeout=(30, 300),
        )
        confirmed.raise_for_status()
        return confirmed

    parser = _DownloadFormParser()
    parser.feed(html)
    if parser.action:
        confirmed = session.get(
            parser.action,
            params=parser.inputs,
            stream=True,
            timeout=(30, 300),
        )
        confirmed.raise_for_status()
        return confirmed

    raise RuntimeError(
        "Google Drive returned an HTML page without a download confirmation "
        f"form. Download {OFFICIAL_SHARE_URL} manually and place "
        f"{TARGET_NAME} in pretrained/."
    )


def _save_response(response: requests.Response, destination: Path) -> None:
    content_type = response.headers.get("content-type", "").lower()
    if "text/html" in content_type:
        raise RuntimeError(
            "Google Drive returned HTML instead of the checkpoint archive. "
            f"Use the official link: {OFFICIAL_SHARE_URL}"
        )
    with destination.open("wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output.write(chunk)


def _extract_or_move(download: Path, output: Path) -> None:
    if zipfile.is_zipfile(download):
        with zipfile.ZipFile(download) as archive:
            candidates = [
                member
                for member in archive.namelist()
                if Path(member).name == TARGET_NAME
                and not member.endswith("/")
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Official archive contains {len(candidates)} copies of "
                    f"{TARGET_NAME}; expected exactly one."
                )
            with archive.open(candidates[0]) as source, output.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    else:
        shutil.move(str(download), str(output))


def _validate_checkpoint(path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise RuntimeError("Downloaded file is not a GMFlow state dictionary")
    required = {
        "backbone.conv1.weight",
        "transformer.layers.0.self_attn.q_proj.weight",
        "feature_flow_attn.q_proj.weight",
    }
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError(
            "Downloaded checkpoint is not the official single-scale GMFlow "
            "model; missing keys: " + ", ".join(missing)
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pretrained") / TARGET_NAME,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.force:
        _validate_checkpoint(output)
        print(f"Checkpoint already present and valid: {output}")
        print(f"SHA256: {_sha256(output)}")
        return

    with tempfile.TemporaryDirectory(
        prefix="gmflow-download-",
        dir=output.parent,
    ) as temporary_directory:
        downloaded = Path(temporary_directory) / "official_download"
        with requests.Session() as session:
            response = _binary_response(session, OFFICIAL_FILE_ID)
            _save_response(response, downloaded)
        temporary_output = Path(temporary_directory) / TARGET_NAME
        _extract_or_move(downloaded, temporary_output)
        _validate_checkpoint(temporary_output)
        temporary_output.replace(output)

    print(f"Downloaded official GMFlow checkpoint: {output}")
    print(f"SHA256: {_sha256(output)}")


if __name__ == "__main__":
    main()
