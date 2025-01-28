import os
import trimesh
import open3d as o3d
import numpy as np
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_random_pcds(modelnet_dir, output_dir, num_pcds=10, num_points=2048):
    """Generates random PCDs from ModelNet10 test set."""

    test_dir = os.path.join(modelnet_dir, "modelnet10_test")

    if not os.path.exists(test_dir):
        logger.error(f"ModelNet10 test directory not found: {test_dir}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    all_files = [] # Initialize the list HERE (no comment) <--- Corrected!

    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                if filename.endswith(".off"):
                    filepath = os.path.join(class_dir, filename)
                    all_files.append(filepath)

    if not all_files:
        logger.error("No.off files found in the ModelNet10 test directory.")
        return

    if len(all_files) < num_pcds:
        logger.warning(f"Only {len(all_files)}.off files available. Generating that many PCDs.")
        num_pcds = len(all_files)

    random_files = random.sample(all_files, num_pcds)

    for i, off_file in enumerate(random_files):
        try:
            mesh = trimesh.load(off_file)
            points = mesh.sample(num_points)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points

            pcd_filename = f"random_point_cloud_{i + 1}.pcd"
            pcd_path = os.path.join(output_dir, pcd_filename)
            o3d.io.write_point_cloud(pcd_path, pcd)
            logger.info(f"PCD saved to: {pcd_path}")

        except Exception as e:
            logger.error(f"Error processing {off_file}: {e}")


if __name__ == "__main__":
    modelnet_dir = "data/ModelNet10"  # Replace with your ModelNet10 directory
    output_dir = "generated_pcds"  # Directory to save the PCDs
    num_pcds = 10  # Number of PCDs to generate.
    generate_random_pcds(modelnet_dir, output_dir, num_pcds=num_pcds)
    print("PCD generation complete.")