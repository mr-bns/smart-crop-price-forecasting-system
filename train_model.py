"""
train_model.py  —  Legacy training entry point (root-level convenience script)
===============================================================================
This script delegates to model/train.py which contains the full training pipeline.
Kept for backward compatibility with older deployment scripts.

Usage:  python train_model.py
"""
import os
import sys

# Ensure project root is on path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from model.train import train_all_models

if __name__ == "__main__":
    train_all_models()
