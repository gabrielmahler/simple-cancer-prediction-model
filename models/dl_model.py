import torch
import torch.nn as nn
import numpy as np

class DLModel(nn.Module):
    def __init__(self):
        super(DLModel,self).__init__()
        self.d1 = nn.Linear(7, 50)
        self.d2 = nn.Linear(50, 200)
        self.relu0 = nn.ReLU()
        self.d3 = nn.Linear(200, 50)
        self.relu1 = nn.ReLU()
        self.d4 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.relu0(x)
        x = self.d3(x)
        x = self.relu1(x)
        x = self.d4(x)
        x = self.sigmoid(x)
        return x

def shuffle_data(data, ratio=0.8):
    np.random.shuffle(data)
    split = int(len(data)*ratio)
    inputs = data[:, :-2]
    labels = data[:, -1:]
    train_inputs = torch.tensor(inputs[:split], dtype=torch.float32)
    train_labels = torch.tensor(labels[:split], dtype=torch.float32)
    test_inputs = torch.tensor(inputs[split:], dtype=torch.float32)
    test_labels = torch.tensor(labels[split:], dtype=torch.float32)
    # print(train_inputs.shape, train_labels.shape, test_inputs.shape, test_labels.shape)
    return train_inputs, train_labels, test_inputs, test_labels


def train(model, all_data, optimizer, criterion, epochs=1000, batch_size=20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        train_inputs, train_labels, test_inputs, test_labels = shuffle_data(all_data)
        for i in range(0, len(train_inputs), batch_size):
            optimizer.zero_grad()
            outputs = model(train_inputs)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        with torch.no_grad():
            outputs = model(test_inputs)
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0
            correct1 = (outputs[:, 0] == test_labels[:, 0]).sum()
            # correct2 = (outputs[:, 1] == test_labels[:, 1]).sum()
            # print(correct1)
        print(f'Epoch {epoch}/{epochs}, Loss: {running_loss}, Accuracy on first variable: {int(correct1/len(test_labels) * 100)}%')
    print('Finished Training')
    torch.save(model.state_dict(), 'models/dl_model.pth')
    print('Model saved')

def load_data(path):
    try:
        data = np.genfromtxt(path, delimiter=',', skip_header=1)
        print(data.shape)
        return data
    except Exception as e:
        print(f'Error loading data: {e}')
        return None


def main():
    data = load_data('data/The_Cancer_data_1500_V2.csv')
    if data is not None:
        print("Data found and loaded")
        model = DLModel()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
        train(model, data, optimizer, criterion)
    else:
        print('No data to train on')

if __name__=='__main__':
    main()