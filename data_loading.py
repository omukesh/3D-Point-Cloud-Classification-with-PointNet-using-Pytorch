import trimesh
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelNetDataset(Dataset):
    def __init__(self, root_dir, split='train', num_classes=10, num_points=2048):
        self.root_dir = root_dir
        self.split = split
        self.num_classes = num_classes
        self.num_points = num_points
        self.classes = sorted(os.listdir(os.path.join(root_dir, f'modelnet10_{self.split}')))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.data = []

        for cls in self.classes:
            class_dir = os.path.join(root_dir, f'modelnet10_{self.split}', cls)
            logger.info(f"Processing class directory: {class_dir}")
            if not os.path.isdir(class_dir):
                logger.error(f"Directory does not exist: {class_dir}")
                continue
            logger.info(f"Contents of {class_dir}: {os.listdir(class_dir)}")
            for filename in os.listdir(class_dir):
                if filename.endswith(".off"):
                    filepath = os.path.join(class_dir, filename)
                    self.data.append((filepath, self.class_to_idx[cls]))
                    logger.info(f"Added file: {filepath}")

        logger.info(f"Loaded {len(self.data)} samples for {self.split} split.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filepath, label = self.data[idx]
        try:
            mesh = trimesh.load(filepath, force='mesh')
            points = mesh.vertices
            if points.shape[0] < self.num_points:
                # Pad with zeros if the number of points is less than num_points
                padding = np.zeros((self.num_points - points.shape[0], 3))
                points = np.concatenate([points, padding], axis=0)
            else:
                # Sample points if the number of points is more than num_points
                points = points[np.random.choice(points.shape[0], self.num_points, replace=False), :]
            points = self.normalize_point_cloud(points)
            return torch.tensor(points, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            logger.error(f"Error loading file {filepath}: {e}")
            # Return a dummy sample to skip this file
            return torch.tensor([[0, 0, 0]], dtype=torch.float32), torch.tensor(0, dtype=torch.long)

    def normalize_point_cloud(self, points):
        centroid = np.mean(points, axis=0)
        points = points - centroid
        furthest_distance = np.max(np.sqrt(np.sum(points**2, axis=-1)))
        points = points / furthest_distance
        return points

def create_dataloaders(root_dir, num_classes=10, num_points=2048, batch_size=32, num_workers=4):
    train_dataset = ModelNetDataset(root_dir, split="train", num_classes=num_classes, num_points=num_points)
    test_dataset = ModelNetDataset(root_dir, split="test", num_classes=num_classes, num_points=num_points)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
