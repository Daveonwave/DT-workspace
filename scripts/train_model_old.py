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


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


# RUN TRAINING TROUGH CROSS-VALIDATION
def run_cross_validation(args):
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

    # Prepare dataset and split in train and test (train for cross validation)
    df = pd.read_csv('models/checkup_dataset.csv')

    x_train = df.drop(['temperature'], axis=1)
    y_train = df['temperature']

    # Standardizing data
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)

    # Tensor format of X and y
    x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train.values, dtype=torch.float32, device=device).reshape(-1, 1)

    # Dataset defined as a torch dataset
    dataset = SimpleDataset(x_train, y_train)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=args.k_folds, shuffle=True)

    # Start print
    print('--------------------------------')

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(dataset, batch_size=len(test_ids), sampler=test_subsampler)

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
        print('Training process has finished. Saving trained model.')
        # Print about testing
        print('Starting testing')

        # Saving the model
        if args.save_model:
            save_model_path = (f'./models/weights/model_fold_{fold}.pth')
            torch.save(model.state_dict(), save_model_path)

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
            print('MSE for fold %d: %f' % (fold, mse))
            print('MaAE for fold %d: %f' % (fold, max_er))
            print('MAPE for fold %d: %f %%' % (fold, mape))
            print('R2_score for fold %d: %f' % (fold, r2))
            print('--------------------------------')
            results[fold] = {'MSE': mse,
                             'MaAE': max_er,
                             'MAPE': mape,
                             'R2_SCORE': r2
                             }

    if args.save_model:
        save_scaler_path = f'./models/weights/k-fold-scaler.pth'
        joblib.dump(scaler, save_scaler_path)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {args.k_folds} FOLDS')
    print('--------------------------------')
    for metric in ['MSE', 'MaAE', 'MAPE', 'R2_SCORE']:
        print('Metric {}:'.format(metric))
        sum = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value[metric]}')
            sum += value[metric]
        print(f'Average: {sum / len(results.items())}')
        print('--------------------------------')


# RUN TRAINING ON THE ENTIRE DATASET
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

    # Prepare dataset and split in train and test (train for cross validation)
    df_train = pd.read_csv('models/checkup_dataset.csv')
    df_train = shuffle(df_train)
    df_test = pd.read_csv('models/pv2_eval.csv')
    df_test = shuffle(df_test)
    # df_train = df.sample(frac=0.8, random_state=200)
    # df_test = df.drop(df_train.index)

    x_train = df_train.drop(['temperature', 'voltage'], axis=1)
    x_test = df_test.drop(['temperature', 'voltage'], axis=1)
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
        save_path = f'./models/weights/model_complete_pv1.pth'
        torch.save(model.state_dict(), save_path)
        save_scaler_path = f'./models/weights/complete_scaler_pv1.pth'
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
        print('MAPE: {} %%'.format(mape))
        print('R2_score: {}'.format(r2))
        print('--------------------------------')


def run_lstm(args):
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

    def create_dataset(dataset, lookback):
        """Transform a time series into a prediction dataset

        Args:
            dataset: A numpy array of time series, first dimension is the time steps
            lookback: Size of window for prediction
        """
        X, y = [], []
        for i in range(len(dataset) - lookback):
            feature = dataset[i:i + lookback]
            target = dataset[i + 1:i + lookback + 1]
            X.append(feature)
            y.append(target)
        return torch.tensor(X), torch.tensor(y)

    df = pd.read_csv('./models/pv1_eval.csv')
    timeseries = df[["temperature"]].values.astype('float32')

    train_size = int(len(timeseries) * 0.67)
    test_size = len(timeseries) - train_size
    train, test = timeseries[:train_size], timeseries[train_size:]

    lookback = 1
    X_train, y_train = create_dataset(train, lookback=lookback)
    X_test, y_test = create_dataset(test, lookback=lookback)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    model = LSTMModel()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

    for epoch in range(args.num_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        #if epoch % 100 != 0:
        #    continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

    with torch.no_grad():
        # shift train predictions for plotting
        train_plot = np.ones_like(timeseries) * np.nan
        train_plot[lookback:train_size] = model(X_train)[:, -1, :]
        # shift test predictions for plotting
        test_plot = np.ones_like(timeseries) * np.nan
        test_plot[train_size + lookback:len(timeseries)] = model(X_test)[:, -1, :]
    # plot
    plt.plot(timeseries, c='b')
    plt.plot(train_plot, c='r')
    plt.plot(test_plot, c='g')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Network Regression Example")
    parser.add_argument("--input_size", type=int, default=4, help="Size of input features")
    parser.add_argument("--hidden_size", type=int, default=16, help="Number of neurons in the hidden layer")
    parser.add_argument("--output_size", type=int, default=1, help="Size of output (regression output)")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for SGD optimizer")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--cross_val", action="store_true", help="Train with k-fold crossvalidation")
    parser.add_argument("--lstm", action="store_true", help="Train with LSTM")
    parser.add_argument("--save_model", action="store_true", help="Save model")
    args = parser.parse_args()

    if args.cross_val:
        run_cross_validation(args)

    elif args.lstm:
        run_lstm(args)

    else:
        run(args)

