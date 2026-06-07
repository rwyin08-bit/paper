#!/usr/bin/env python3
"""
Download and prepare SPE 10 Comparative Solution Project dataset.

Dataset: 60x220x85 geostatistical grid (1,122,000 cells)
Reference: Christie, M.A., Blunt, M.J. (2001). SPE Reservoir Evaluation & Engineering 4(4), 308-317.
Source: https://www.spe.org/web/csp/datasets/set02.htm
"""

import os
import sys
import urllib.request
import tarfile

SPE10_URL = "https://www.spe.org/web/csp/datasets/spe10.tar.gz"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def download_with_progress(url, dest):
    print(f"Downloading SPE 10 dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
    except Exception as e:
        print(f"\nError downloading: {e}")
        print("Please download manually from: https://www.spe.org/web/csp/datasets/set02.htm")
        print(f"and extract to: {DATA_DIR}")
        sys.exit(1)

def _progress(count, block_size, total_size):
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write(f"\r  {percent}%")
    sys.stdout.flush()

def extract_tar(tar_path, dest):
    print("\nExtracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=dest)
    os.remove(tar_path)
    print("Done.")

def verify():
    expected = ["spe10_perm.dat", "spe10_phi.dat"]
    for f in expected:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            print(f"  [OK] {f} ({os.path.getsize(path)/(1024*1024):.1f} MB)")
        else:
            print(f"  [MISSING] {f}")

def main():
    if os.path.exists(os.path.join(DATA_DIR, "spe10_perm.dat")):
        print("SPE 10 dataset already exists.")
        verify()
        return
    tar_path = os.path.join(DATA_DIR, "spe10.tar.gz")
    download_with_progress(SPE10_URL, tar_path)
    extract_tar(tar_path, DATA_DIR)
    print("\nVerification:")
    verify()

if __name__ == "__main__":
    main()
