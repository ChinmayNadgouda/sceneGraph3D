import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.nn import global_max_pool
import torch.optim as optim
import os
import open3d as o3d
import numpy as np
import torch
# # from torch.utils.data import Dataset

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
        self.input_transform = TNet(3)

        self.feature_transform = TNet(64)

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def forward(self, data):
        x, batch = data.x, data.batch

        # Input transform
        x = x.view(batch.size(0), 3, -1)
        trans_input = self.input_transform(x)
        x = torch.bmm(trans_input, x)
        x = self.relu(self.bn1(self.conv1(x)))
        print('1')

        # Feature transform
        trans_feat = self.feature_transform(x)
        x = torch.bmm(trans_feat, x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        print('2')

        # Global max pooling
        x = global_max_pool(x.squeeze() , batch.squeeze() )
        print('3')

        # Fully connected layers
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        return F.log_softmax(x, dim=-1)


class PointCloudDataset(Dataset):
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

        # Open and read the text file
        with open(label_file, 'r') as file:
            for line in file:
                # Split each line by comma to get id and label
                id_, label = line.strip().split(',')
                # Store in dictionary with id as key and label as value
                data_dict[id_] = int(label)

        # Sort the dictionary by id keys
        sorted_data_dict = dict(sorted(data_dict.items()))

        self.labels = [int(label) for label in list(sorted_data_dict.values())]
        

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
        label = self.labels[idx]

        pos = torch.tensor(points, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        data = Data(x=pos, y=y, pos=pos)  # Use pos for both x and pos (features and positions)

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
criterion = nn.CrossEntropyLoss()

# Training loop
def train(model, loader, criterion, optimizer, device):
    model.train()
    for data in loader:
        data = data.to(device)
        
        optimizer.zero_grad()
        print("Batch shape:", data)
        print("Batch.pos shape:", data.pos.shape)
        out = model(data)
        _, predicted = torch.max(out.data, 1) 
        print('Input', data)
        print('Output',predicted)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

# Device setup (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(num_epochs):
    train(model, dataloader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}/{num_epochs} complete")
#torch.save(model, "3Dpointcloudnetwork.pth")

# Example Testing usage
ply_dir_test = '/home/student/test_without_exclude_class'
label_file_test = '/home/student/test_without_exclude_class/label.txt'
test_dataset = PointCloudDataset(ply_dir_test, label_file_test)
# Assuming you have a test DataLoader named `test_loader`
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # test_dataset should be the test data

#Load the entire model directly
model = torch.load("3Dpointcloudnetwork.pth")

# Switch to evaluation mode if using for testing/inference
def test(model, loader, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient calculations for inference
        for data in loader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)  # Get the index of the highest score as the predicted class
            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# Run the test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test(model, test_loader, device)