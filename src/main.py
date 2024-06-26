import argparse
from src.train.train import load_model, train

def main():
    parser = argparse.ArgumentParser(description="Train a YOLO model with specified parameters.")
    parser.add_argument('--model_id', type=str, required=True, help='Path to the YOLO model file.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the training data.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--imgsz', type=int, default=416, help='Image size for training.')
    parser.add_argument('--batch', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')

    args = parser.parse_args()

    model = load_model(args.model_id)
    results = train(model, args.data_path, args.epochs, args.imgsz, args.batch, args.verbose)
    
    print("Training results:", results)


if __name__ == "__main__" : 
    main()

    