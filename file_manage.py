import os
import shutil

# Define the source and destination directories
source_dir = '/home/mukesh/Downloads/archive/ModelNet10'
train_dest_dir = os.path.join(source_dir, 'home/PycharmProjects/point_cloud/pcl_processing/data/ModelNet10/modelnet10_train')
test_dest_dir = os.path.join(source_dir, 'home/PycharmProjects/point_cloud/pcl_processing/data/ModelNet10/modelnet10_test')

# List of classes
classes = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

# Ensure the destination directories exist
os.makedirs(train_dest_dir, exist_ok=True)
os.makedirs(test_dest_dir, exist_ok=True)

# Move the train and test folders
for cls in classes:
    class_dir = os.path.join(source_dir, cls)
    print(f"Processing class: {cls}")
    print(f"Class directory: {class_dir}")

    # List all items in the class directory
    items = os.listdir(class_dir)
    print(f"Items in {cls} directory: {items}")

    train_src = os.path.join(class_dir, 'train')
    test_src = os.path.join(class_dir, 'test')

    train_dest = os.path.join(train_dest_dir, cls)
    test_dest = os.path.join(test_dest_dir, cls)

    # Move the train folder
    if os.path.exists(train_src):
        shutil.move(train_src, train_dest)
        print(f"Moved {train_src} to {train_dest}")
    else:
        print(f"{train_src} does not exist")

    # Move the test folder
    if os.path.exists(test_src):
        shutil.move(test_src, test_dest)
        print(f"Moved {test_src} to {test_dest}")
    else:
        print(f"{test_src} does not exist")

print("All folders moved successfully.")
