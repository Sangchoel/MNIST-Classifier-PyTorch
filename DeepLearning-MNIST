import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = (shape,)

    def forward(self, x):
        return x.view(*self.shape)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, stride=2),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, kernel_size=3, stride=2),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(10),
            View(250),
            nn.Linear(250, 10),
            nn.Sigmoid()
        )
        self.loss_function = nn.BCELoss()
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.001)
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        self.progress.append(loss.item())

    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))

class MnistDataset(Dataset):
    def __init__(self, csv_file):
        self.data_df = pd.read_csv(csv_file, header=None)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        label = self.data_df.iloc[index, 0]
        target = torch.zeros(10)
        target[label] = 1.0
        image_values = torch.FloatTensor(self.data_df.iloc[index, 1:].values) / 255.0
        return label, image_values.view(1, 28, 28), target

    def plot_image(self, index):
        img = self.data_df.iloc[index, 1:].values.reshape(28, 28)
        plt.title(f'label = {self.data_df.iloc[index, 0]}')
        plt.imshow(img, interpolation='none', cmap='Blues')
        plt.show()

mnist_dataset = MnistDataset('/content/drive/MyDrive/AI_2023/sample_data/mnist_train.csv')
mnist_test_dataset = MnistDataset('/content/drive/MyDrive/AI_2023/sample_data/mnist_test.csv')
classifier = Classifier()

# 훈련 과정
epochs = 4
for i in range(epochs):
    print(f'Training epoch {i+1} of {epochs}')
    for label, image_data_tensor, target_tensor in mnist_dataset:
        # 배치 차원 추가
        classifier.train(image_data_tensor.unsqueeze(0), target_tensor)


classifier.plot_progress()

# 테스트 과정
items = 0
score = 0

for label, image_data_tensor, target_tensor in mnist_test_dataset:
    # 배치 차원을 추가하여 모델 입력 차원에 맞춤
    output = classifier.forward(image_data_tensor.unsqueeze(0))
    if output.detach().numpy().argmax() == label:
        score += 1
    items += 1


print(score, items, score/items)


print(f'Accuracy: {score/items:.2f}')
