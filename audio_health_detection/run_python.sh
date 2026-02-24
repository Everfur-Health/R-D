#!/usr/bin/env bash

source ./venv/bin/activate
cd ./python

python organize_data.py --input ../data/raw --output ../data/processed
python augment_data.py --input ../data/processed --output ../data/augmented
python train.py
python web/server.py

