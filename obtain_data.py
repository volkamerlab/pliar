#!/usr/bin/env python3
"""
Download and extract all ZIP archives from a Zenodo record (streamed to disk).

Usage:
    python download_zenodo_zips_stream.py <zenodo_url> <target_directory>

Example:
    python download_zenodo_zips_stream.py https://zenodo.org/record/1234567 .
"""

import os
import re
from typing import Optional
import requests
from zipfile import ZipFile
import typer


def get_zenodo_record_id(url: str) -> str:
    """Extract the record ID from a Zenodo URL."""
    match = re.search(r"/(?:record|records)/(\d+)", url)
    if not match:
        raise ValueError("Could not extract Zenodo record ID from URL.")
    return match.group(1)


def get_zenodo_files(record_id: str):
    """Fetch metadata from the Zenodo API and return all .zip file URLs."""
    api_url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(api_url)
    response.raise_for_status()
    data = response.json()

    files = data.get("files", [])
    zip_files = [
        {"name": f["key"], "url": f["links"]["self"]}
        for f in files
        if f["key"].endswith(".zip")
    ]
    return zip_files


def download_zip_to_disk(zip_info, target_dir):
    """Download a single ZIP file to disk."""
    name = zip_info["name"]
    url = zip_info["url"]
    local_path = os.path.join(target_dir, name)

    print(f"⬇️  Downloading {name}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return local_path


def extract_zip(file_path, target_dir):
    """Extract a ZIP file to the target directory."""
    print(f"📦 Extracting {os.path.basename(file_path)}...")
    with ZipFile(file_path, "r") as zf:
        zf.extractall(target_dir)
    print(f"✅ Extracted {os.path.basename(file_path)}")


def main(zenodo_url: str, target_dir: str = ".", single_file: Optional[str] = None):
    if not os.path.exists(target_dir):
        raise FileNotFoundError(target_dir)

    record_id = get_zenodo_record_id(zenodo_url)
    print(f"Fetching ZIP file list for record {record_id}...")

    zip_files = get_zenodo_files(record_id)
    if not zip_files:
        print("No .zip files found in this Zenodo record.")
        return

    print(f"Found {len(zip_files)} zip file(s).")

    if single_file is not None:
        print(f"\nSearching for a single file {single_file} in {zenodo_url}")
        zip_files = [item for item in zip_files if item["name"] == single_file]
        match len(zip_files):
            case 0:
                print("No such file found! Exiting.")
                exit(0)
            case 1:
                print("File found! Continuing...")
            case _:
                raise RuntimeError(f"Found duplicate file {single_file}!")

    for zip_info in zip_files:
        file_path = download_zip_to_disk(zip_info, target_dir)
        extract_zip(file_path, target_dir)
        # Optionally delete the zip file after extraction
        os.remove(file_path)

    print(f"\n🎉 All files extracted into: {os.path.abspath(target_dir)}")


if __name__ == "__main__":
    typer.run(main)
