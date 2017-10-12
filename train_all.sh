#!/bin/sh

# Train all three models
python train_vqg.py --num_epochs=3
python train_qa.py --num_epochs=3
python train_vqa.py --num_epochs=3
