import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os
import logging
from data_loading import create_dataloaders  # Import from data_loader.py
from model import PointNet  # Import from model.py

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

if not os.path.exists("models"):
    os.makedirs("models")

def train_model(root_dir, num_classes=10, num_points=2048, num_epochs=50, learning_rate=0.001, patience=10):
    train_loader, test_loader = create_dataloaders(root_dir, num_classes=num_classes, num_points=num_points)
    model = PointNet(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for points, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            if points is None or labels is None:
                continue
            points = points.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation (or testing)
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for points, labels in test_loader:
                if points is None or labels is None:
                    continue
                points = points.to(device)
                labels = labels.to(device)
                outputs = model(points)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds)
        logger.info(f"Epoch {epoch + 1} Validation Accuracy: {accuracy}")

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            torch.save(model.state_dict(), "models/pointnet_model.pth")
            logger.info("Model saved!")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

if __name__ == '__main__':
    train_model("data/ModelNet10", num_classes=10, num_points=2048, num_epochs=50)
