import torch
import open3d as o3d
import numpy as np
import logging
import argparse
import os
from data_loading import ModelNetDataset  # Make sure data_loading.py is in the same directory or path is correct
from model import PointNet  # Make sure model.py is in the same directory or path is correct

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def infer(model_path, point_cloud_path, root_dir):
    """
    Performs inference with the trained PointNet model.
    """

    if not os.path.exists(point_cloud_path):
        logger.error(f"Point cloud file not found: {point_cloud_path}")
        return

    try:
        train_dataset = ModelNetDataset(root_dir, split='train')  # For preprocessing and class names
        num_classes = len(train_dataset.classes) # Get the number of classes dynamically
        model = PointNet(num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device)) #map_location is important
        model.eval()

        pcd = o3d.io.read_point_cloud(point_cloud_path)

        if pcd.is_empty():
            logger.error("Loaded point cloud is empty. Check the file.")
            return

        points = np.asarray(pcd.points)

        if points.size == 0: # Check if points array is empty
            logger.error("No points found in PCD file")
            return

        points = train_dataset.normalize_point_cloud(points)  # Important: Same normalization as training
        points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            outputs = model(points_tensor)
            _, predicted_class = torch.max(outputs, 1)

        predicted_class_name = train_dataset.classes[predicted_class.item()]
        logger.info(f"Predicted class: {predicted_class_name}")

        # Visualize the point cloud with the prediction in the title
        o3d.visualization.draw_geometries([pcd], window_name=f"Predicted Class: {predicted_class_name}")

    except Exception as e:
        logger.error(f"Error during inference: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform inference with PointNet model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--point_cloud_path", type=str, required=True, help="Path to the point cloud file.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of the dataset.")
    args = parser.parse_args()

    infer(args.model_path, args.point_cloud_path, args.root_dir)