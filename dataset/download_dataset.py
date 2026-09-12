"""
Automated Dataset Ingestion Script for Google Colab / Local Setup
-----------------------------------------------------------------
Downloads the Roboflow Universe Construction Site Safety / PPE Dataset.

Usage:
    python download_dataset.py --api-key YOUR_ROBOFLOW_API_KEY
    OR use the direct public download mirror provided.
"""

import os
import argparse
import urllib.request
import zipfile
from pathlib import Path

def download_via_roboflow(api_key: str, dest_dir: str):
    """
    Downloads dataset via official Roboflow API.
    Project: construction-site-safety / PPE Detection
    Classes: hard-hat, safety-vest, person
    """
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        # Using the standard benchmark Construction Site Safety dataset
        project = rf.workspace("roboflow-universe-projects").project("construction-site-safety")
        version = project.version(1)
        dataset = version.download("yolov8", location=dest_dir)
        print(f"Dataset successfully downloaded to: {dest_dir}")
    except ImportError:
        print("Error: roboflow package not installed. Run: pip install roboflow")
    except Exception as e:
        print(f"Roboflow download encountered an error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Construction PPE Dataset")
    parser.add_argument("--api-key", type=str, default="", help="Roboflow API Key (optional)")
    parser.add_argument("--dest", type=str, default=".", help="Destination directory")
    args = parser.parse_args()

    if args.api_key:
        download_via_roboflow(args.api_key, args.dest)
    else:
        print("No API key provided. Refer to dataset/README.md for direct public archive links and Colab commands.")
