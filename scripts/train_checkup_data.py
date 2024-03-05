import argparse
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, max_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# Define the neural network model
class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)


# RUN TRAINING ON CHECKUP DATASET WITH MOVING AVERAGE
def run(args):
    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "mps" if torch.backends.mps.is_available()
                                                                        and args.cuda else "cpu")
    # Define loss function
    loss_function = nn.MSELoss()

    # For fold results
    results = {}

    # Generate some random data
    torch.manual_seed(args.seed)  # For reproducibility
    
    # Columns to consider
    columns = ['current', 'heat', 't_amb', 'rolling_25', 'temperature']

    # Prepare dataset and split in train and test (train for cross validation)
    df_train = pd.read_csv('./datasets/full_train.csv')
    df_train = df_train[columns]
    df_train = df_train.dropna()
    df_train = shuffle(df_train)
    
    df_test = pd.read_csv('./datasets/train_pv1.csv')
    df_test = df_test[columns]
    df_test = df_test.dropna()
    # df_train = df.sample(frac=0.8, random_state=200)
    # df_test = df.drop(df_train.index)

    x_train = df_train.drop(['temperature'], axis=1)
    x_test = df_test.drop(['temperature'], axis=1)
    y_train = df_train['temperature']
    y_test = df_test['temperature']

    # Standardizing data
    scaler = MinMaxScaler()
    scaler.fit(x_train.values)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Tensor format of X and y
    x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train.values, dtype=torch.float32, device=device).reshape(-1, 1)
    x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test.values, dtype=torch.float32, device=device).reshape(-1, 1)

    # Dataset defined as a torch dataset
    train_dataset = SimpleDataset(x_train, y_train)
    test_dataset = SimpleDataset(x_test, y_test)

    #dataset = ConcatDataset([train_dataset, test_dataset])

    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=10)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))

    # Create an instance of the model
    model = RegressionModel(args.input_size, args.hidden_size, args.output_size).to(device)
    model.apply(reset_weights)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Run the training loop for defined number of epochs
    for epoch in range(args.num_epochs):

        # Print epoch
        print(f'Starting epoch {epoch + 1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):

            # Get inputs
            inputs, targets = data
            # Zero the gradients
            optimizer.zero_grad()
            # Perform forward pass
            outputs = model(inputs)
            # Compute loss
            loss = loss_function(outputs, targets)
            # Perform backward pass
            loss.backward()
            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500))
                current_loss = 0.0

    # Process is complete.
    print('Training process has finished.')

    # Saving the model
    if args.save_model:
        print('Saving trained model.')
        save_path = './models/weights/state_{}.pth'.format(args.save_model)
        torch.save(model.state_dict(), save_path)
        save_scaler_path = './models/scaler/scaler_{}.pth'.format(args.save_model)
        joblib.dump(scaler, save_scaler_path)

    # Print about testing
    print('Starting testing')

    # Evaluation for this fold
    with torch.no_grad():
        # Iterate over the test data and generate predictions
        for i, data in enumerate(testloader, 0):
            # Get inputs
            inputs, targets = data
            # Generate outputs
            outputs = model(inputs)

            # Compute metrics
            mse = mean_squared_error(targets, outputs)
            max_er = max_error(targets, outputs)
            mape = mean_absolute_percentage_error(targets, outputs) * 100
            r2 = r2_score(targets, outputs)

        # Print metrics
        print('--------------------------------')
        print('EVALUATION...')
        print('--------------------------------')
        print('MSE: {}'.format(mse))
        print('MaAE: {}'.format(max_er))
        print('MAPE: {} %'.format(mape))
        print('R2_score: {}'.format(r2))
        print('--------------------------------')
