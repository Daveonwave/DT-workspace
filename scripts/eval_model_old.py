import argparse

import joblib
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, max_error


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
            #nn.ReLU(),
            #nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)


def evaluate(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "mps" if torch.backends.mps.is_available()
                                                                        and args.cuda else "cpu")
    # For fold results
    results = {}

    # Generate some random data
    torch.manual_seed(args.seed)  # For reproducibility

    # Prepare dataset and split in train and test (train for cross validation)
    df = pd.read_csv('models/checkup_dataset.csv')

    x_test = df.drop(['temperature'], axis=1)
    y_test = df['temperature']

    # Standardizing data
    scaler = joblib.load('./models/weights/complete-scaler.pth')
    x_test = scaler.transform(x_test)

    # Tensor format of X and y
    x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test.values, dtype=torch.float32, device=device).reshape(-1, 1)

    test_dataset = SimpleDataset(x_test, y_test)

    test_subsampler = torch.utils.data.SubsetRandomSampler([0, len(test_dataset)])

    # Define data loaders for training and testing data in this fold
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))

    # Create an instance of the model
    model = RegressionModel(args.input_size, args.hidden_size, args.output_size).to(device)

    for fold in range(5):
        model.load_state_dict(torch.load("./models/weights/model-fold-{}.pth".format(fold)))

        # Evaluation for this fold
        with torch.no_grad():
            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs, targets = data
                # Generate outputs
                outputs = model(inputs)
                # Compute loss
                mse = mean_squared_error(targets, outputs)
                max_er = max_error(targets, outputs)
                mape = mean_absolute_percentage_error(targets, outputs) * 100
                r2 = r2_score(targets, outputs)

            # Print metrics
            print('MSE for fold %d: %d' % (fold, mse))
            print('MaAE for fold %d: %d' % (fold, max_er))
            print('MAPE for fold %d: %d %%' % (fold, mape))
            print('R2_score for fold %d: %d' % (fold, r2))
            print('--------------------------------')
            results[fold] = {'MSE': mse,
                             'MaAE': max_er,
                             'MAPE': mape,
                             'R2_SCORE': r2
                             }

    print(f'RESULTS OF MODELS GET BY {args.k_folds} FOLDS')
    print('--------------------------------')
    for metric in ['MSE', 'MaAE', 'MAPE', 'R2_SCORE']:
        print('Metric {}:'.format(metric))
        sum = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value[metric]}')
            sum += value[metric]
        print(f'Average: {sum / len(results.items())}')
        print('--------------------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Network Regression Example")
    parser.add_argument("--input_size", type=int, default=5, help="Size of input features")
    parser.add_argument("--hidden_size", type=int, default=16, help="Number of neurons in the hidden layer")
    parser.add_argument("--output_size", type=int, default=1, help="Size of output (regression output)")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--save_model", action="store_true", help="Save model")
    args = parser.parse_args()

    evaluate(args)
