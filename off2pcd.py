import torch
import open3d as o3d
import numpy as np
import logging
import argparse
import os
import trimesh  # Import trimesh for.off handling
from data_loading import ModelNetDataset
from model import PointNet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def infer(model_path, off_path, root_dir):  # Changed point_cloud_path to off_path
    """Performs inference with the trained PointNet model on an OFF file."""

    if not os.path.exists(off_path):  # Check for OFF file existence
        logger.error(f"OFF file not found: {off_path}")
        return

    try:
        train_dataset = ModelNetDataset(root_dir, split='train')
        num_classes = len(train_dataset.classes)
        model = PointNet(num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Load and convert OFF to PCD
        mesh = trimesh.load(off_path)
        points = mesh.sample(2048)  # Sample points from the mesh (adjust num_points if needed)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if pcd.is_empty():
            logger.error("Converted point cloud is empty. Check the OFF file.")
            return

        points = np.asarray(pcd.points)
        points = train_dataset.normalize_point_cloud(points)
        points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(points_tensor)
            _, predicted_class = torch.max(outputs, 1)

        predicted_class_name = train_dataset.classes[predicted_class.item()]
        logger.info(f"Predicted class: {predicted_class_name}")

        o3d.visualization.draw_geometries([pcd], window_name=f"Predicted Class: {predicted_class_name}")

    except Exception as e:
        logger.error(f"Error during inference: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform inference with PointNet model on an OFF file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--off_path", type=str, required=True, help="Path to the OFF file.") # Changed to --off_path
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of the dataset.")
    args = parser.parse_args()

    infer(args.model_path, args.off_path, args.root_dir) # Changed to args.off_path