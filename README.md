# 3D Point Cloud Classification with PointNet using PyTorch

This repository contains the implementation of a 3D point cloud classification model using PointNet and PyTorch. The model is trained on the ModelNet10 dataset.

## Project Structure

- `data_loading.py`: Script for loading and preprocessing the ModelNet10 dataset.
- `model.py`: Implementation of the PointNet model.
- `training.py`: Script for training the PointNet model.
- `infer.py`: Script for performing inference with the trained PointNet model.
- `pcd.py` : generates point clouds from random test data

## Folder Structure

- data
    - ModelNet
        - modelnet10_test
        - modelnet10_train
- Models
- generated_pcds


## Requirements

- Python 3.x
- PyTorch
- Open3D
- NumPy
- Scikit-learn
- tqdm

## Training

- To train the PointNet model, run the following command:

      python training.py 


This project is for reference purpose and is licensed under Apache 2.0 Licence

- To run inferences on the model generated, 

      python infer.py --model_path models/trained_model.pth --point_cloud_path generated_pcds/point_cloud.pcd --root_dir data/ModelNet10
