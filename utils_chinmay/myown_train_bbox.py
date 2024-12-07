import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.nn import global_max_pool, MessagePassing
import torch.optim as optim
import os
import open3d as o3d
import numpy as np
import torch
# # from torch.utils.data import Dataset
# Custom MLP-based Message Passing Layer
class PointNetConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(PointNetConv, self).__init__(aggr='max')
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),  # in_channels + 3 for x_j + relative_pos
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x, pos, batch):
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=x.device)  # No explicit edges
        return self.propagate(edge_index, x=x, pos=pos, batch=batch)

    def message(self, x_j, pos_j, pos_i):
        relative_pos = pos_j - pos_i
        return self.mlp(torch.cat([x_j, relative_pos], dim=-1))
class TNet(nn.Module):
    def __init__(self, k):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # Initialize the final FC layer as an identity transformation
        nn.init.constant_(self.fc3.weight, 0)
        nn.init.eye_(self.fc3.bias.view(k, k))

    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        return x.view(batch_size, self.k, self.k)

class PointNet(nn.Module):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()
        self.conv1 = PointNetConv(3, 128)
        self.conv2 = PointNetConv(128, 256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.label_head = nn.Linear(128, num_classes)
        self.bbox_head = nn.Linear(128, 6)

    def forward(self, data):
        x, pos, batch = data.pos, data.pos, data.batch

        x = self.conv1(x, pos, batch)
        x = self.conv2(x, pos, batch)
        x = global_max_pool(x, batch)

        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))

        labels = self.label_head(x)
        bbox = self.bbox_head(x)
        return labels, bbox

# Dataset class (already provided)
class PointCloudDataset(Dataset):
    def __init__(self, ply_dir, label_file, num_points=200):
        self.ply_dir = ply_dir
        self.num_points = num_points
        self.ply_files = sorted([f for f in os.listdir(ply_dir) if f.endswith('.ply')])

        self.labels = []
        with open(label_file, 'r') as file:
            for line in file:
                label, min_coords, max_coords = line.strip().split(',')
                label = int(label.strip())
                min_coords = list(map(float, min_coords.strip()[1:-1].split()))
                max_coords = list(map(float, max_coords.strip()[1:-1].split()))
                self.labels.append((min_coords + max_coords, label))

        assert len(self.ply_files) == len(self.labels), "Number of labels must match number of .ply files"

    def __len__(self):
        return len(self.ply_files)

    def __getitem__(self, idx):
        ply_path = os.path.join(self.ply_dir, self.ply_files[idx])
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)

        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
        elif len(points) < self.num_points:
            padding = np.zeros((self.num_points - len(points), 3))
            points = np.vstack((points, padding))

        points = points - np.mean(points, axis=0)
        points = points / np.linalg.norm(points, axis=0)

        bbox, label = self.labels[idx]

        pos = torch.tensor(points, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        bbox = torch.tensor(bbox, dtype=torch.float32).view(-1)
        data = Data(pos=pos, y=y, bbox=bbox)

        return data
class PointCloudDataset2(Dataset):
    def __init__(self, ply_dir, label_file, num_points=200):
        """
        Initialize the dataset.
        
        :param ply_dir: Directory containing the .ply files.
        :param label_file: Path to the text file containing labels.
        :param num_points: Number of points to sample from each point cloud.
        """
        self.ply_dir = ply_dir
        self.num_points = num_points
        self.ply_files = sorted([f for f in os.listdir(ply_dir) if f.endswith('.ply')])

        # Load labels from the label file
        # with open(label_file, 'r') as f:
        #     self.labels = [int(line.strip()) for line in f.readlines()]
        data_dict = {}
        self.labels = []
        # Open and read the text file
        with open(label_file, 'r') as file:
            for line in file:
                # Split each line by comma to get id and label
                label, min, max = line.strip().split(',')
                label = int(label.strip())

                min_cords = list(map(float, min.strip()[1:-1].split()))
                max_cords = list(map(float, max.strip()[1:-1].split()))
                # Store in dictionary with id as key and label as value
                self.labels.append((min_cords+max_cords, label))
        

        assert len(self.ply_files) == len(self.labels), "Number of labels must match number of .ply files"

    def __len__(self):
        return len(self.ply_files)

    def __getitem__(self, idx):
        # Load the point cloud file
        ply_path = os.path.join(self.ply_dir, self.ply_files[idx])
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)

        # Sample or pad the points to the fixed size
        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
        elif len(points) < self.num_points:
            padding = np.zeros((self.num_points - len(points), 3))
            points = np.vstack((points, padding))

        # Normalize the point cloud for consistency
        points = points - np.mean(points, axis=0)
        points = points / np.linalg.norm(points, axis=0)

        # Retrieve label
        bbox, label = self.labels[idx]
    

        pos = torch.tensor(points, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        bbox = torch.tensor(bbox, dtype=torch.float32).view(-1) 
        data = Data(x=pos, y=y, pos=pos, bbox=bbox)  # Use pos for both x and pos (features and positions)

        # Apply any transform if provided
        # if self.transform:
        #     data = self.transform(data)
        
        return data

# Example Training usage
ply_dir = '/home/student/train_without_exclude_class'
label_file = '/home/student/train_without_exclude_class//label.txt'
dataset = PointCloudDataset(ply_dir, label_file)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Iterate through the dataset to get batches of point clouds and labels
#for data in dataloader:
    # print(f"Point Clouds Batch Shape: {point_clouds.shape}")
    # print(f"Labels Batch Shape: {labels.shape}")
    # Here, point_clouds is of shape [batch_size, num_points, 3]
    # labels is of shape [batch_size]
    # Feed these into your neural network for training
# Example usage:
num_classes = 10  # Set the number of classes for classification
model = PointNet(num_classes)

# Hyperparameters
num_epochs = 20
learning_rate = 0.001

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion_class = nn.CrossEntropyLoss()
criterion_bbox = nn.SmoothL1Loss()
# Training loop
def train(model, loader, criterion_class, criterion_bbox, optimizer):
        model.train()
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            batch = batch.to(device)
            labels_pred, bbox_pred = model(batch)

            loss_cls = criterion_class(labels_pred, batch.y)
            loss_bbox = criterion_bbox(bbox_pred, batch.bbox)
            loss = loss_cls + loss_bbox

            loss.backward()
            optimizer.step()
            total_loss += loss.item()



# Device setup (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion_class = nn.CrossEntropyLoss()
criterion_bbox = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    train(model, dataloader, criterion_class, criterion_bbox, optimizer)
    print(f"Epoch {epoch+1}/{num_epochs} complete")
torch.save(model, "3DpointcloudnetworkBbox.pth")

# Example Testing usage
# ply_dir_test = '/home/student/test_without_exclude_class'
# label_file_test = '/home/student/test_without_exclude_class/label.txt'
# test_dataset = PointCloudDataset(ply_dir_test, label_file_test)
# # Assuming you have a test DataLoader named `test_loader`
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # test_dataset should be the test data

# #Load the entire model directly
# model = torch.load("3Dpointcloudnetwork.pth")

# # Switch to evaluation mode if using for testing/inference
# def test(model, loader, device):
#     model.eval()  # Set model to evaluation mode
#     correct = 0
#     total = 0
    
#     with torch.no_grad():  # Disable gradient calculations for inference
#         for data in loader:
#             data = data.to(device)
#             outputs = model(data)
#             _, predicted = torch.max(outputs, 1)  # Get the index of the highest score as the predicted class
#             total += data.y.size(0)
#             correct += (predicted == data.y).sum().item()
    
#     accuracy = 100 * correct / total
#     print(f'Test Accuracy: {accuracy:.2f}%')

# # Run the test
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# test(model, test_loader, device)