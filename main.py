from scripts.train_checkup_data import run
from scripts.train_model_old import run_cross_validation, run_lstm
import argparse

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
    parser.add_argument("--save_model", type=str, default="", help="Save model with name")
    args = parser.parse_args()

    if args.cross_val:
        run_cross_validation(args)

    elif args.lstm:
        run_lstm(args)

    else:
        run(args)