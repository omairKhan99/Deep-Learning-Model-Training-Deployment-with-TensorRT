import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import tensorrt as trt
import numpy as np
import time
import onnx
import onnxruntime as ort

class SimpleCNN(nn.Module): # I am just defining a simple CNN for object detection
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
transform = transforms.Compose([ # I am just defining a simple transform for the CIFAR10 dataset
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

def train_model(): # I am just training the model and exporting it to ONNX
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    
    torch.save(model.state_dict(), "model.pth")
    torch.onnx.export(model, torch.randn(1, 3, 32, 32), "model.onnx")
    print("Model training and ONNX export complete.")

def convert_to_tensorrt(): # I am just converting the ONNX model to TensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open("model.onnx", "rb") as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX model.")
            return
    
    serialized_network = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_network)
    
    with open("model.trt", "wb") as f:
        f.write(serialized_network)
    print("TensorRT model conversion complete.")

def run_onnx_inference(): # I am just running inference using ONNX Runtime
    ort_session = ort.InferenceSession("model.onnx")
    test_input = np.random.rand(1, 3, 32, 32).astype(np.float32)
    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: test_input})
    print("ONNX Runtime Inference Output:", outputs)

if __name__ == "__main__":
    train_model()
    convert_to_tensorrt()
    run_onnx_inference()
