import argparse
from src.train.train import load_model, train
from src.test.test import test_model

def main():
    parser = argparse.ArgumentParser(description="Train or test a YOLO model with specified parameters.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help='Mode to run: train or test.')
    parser.add_argument('--model_id', type=str, required=True, help='Path to the YOLO model file.')

    # Arguments for training
    parser.add_argument('--data_path', type=str, help='Path to the training data.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--imgsz', type=int, default=416, help='Image size for training.')
    parser.add_argument('--batch', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')

    # Arguments for testing
    parser.add_argument('--test_images_dir', type=str, help='Path to the test images directory.')
    parser.add_argument('--results_dir', type=str, help='Directory to save the test results.')

    args = parser.parse_args()

    if args.mode == 'train':
        if not args.data_path:
            print("Data path is required for training.")
            return
        model = load_model(args.model_id)
        results = train(model, args.data_path, args.epochs, args.imgsz, args.batch, args.verbose)
        print("Training results:", results)
    elif args.mode == 'test':
        if not args.test_images_dir or not args.results_dir:
            print("Test images directory and results directory are required for testing.")
            return
        test_model(args.model_id, args.test_images_dir, args.results_dir)

if __name__ == "__main__":
    main()

