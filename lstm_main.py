import os
import sys
import argparse
import json
import torch
import random
from datetime import datetime
from pathlib import Path

from lstm import ClimbGenerator

def setup_parser():
    parser = argparse.ArgumentParser(
        description="Train and generate climbing sequences using adapted LSTM model"
    )
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument(
        "--data", 
        required=True, 
        help="Path to JSON file with climbing data"
    )
    train_parser.add_argument(
        "--epochs", 
        type=int, 
        default=30, 
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Batch size for training"
    )
    train_parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=0.001, 
        help="Learning rate for optimizer"
    )
    train_parser.add_argument(
        "--output", 
        default="lstm_model.pth", 
        help="Path to save trained model"
    )
    

    gen_parser = subparsers.add_parser("generate", help="Generate climbing sequences")
    gen_parser.add_argument(
        "--model", 
        required=True, 
        help="Path to trained model file (.pth)"
    )
    gen_parser.add_argument(
        "--data", 
        required=True, 
        help="Path to JSON file with climbing data"
    )
    gen_parser.add_argument(
        "--count", 
        type=int, 
        default=5, 
        help="Number of sequences to generate"
    )
    gen_parser.add_argument(
        "--climb-id", 
        help="Specific climb ID to generate for (random if not specified)"
    )
    gen_parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.8, 
        help="Sampling temperature (higher = more random)"
    )
    gen_parser.add_argument(
        "--max-length",
        type=int,
        default=30,
        help="Maximum length of generated sequences"
    )
    gen_parser.add_argument(
        "--output-dir", 
        default="generated", 
        help="Directory to save generated sequences"
    )
    gen_parser.add_argument(
        "--export-json", 
        action="store_true", 
        help="Export sequences as JSON file"
    )
    
    #Visualize mode
    vis_parser = subparsers.add_parser("visualize", help="Visualize climbing sequences")
    vis_parser.add_argument(
        "--model", 
        required=True, 
        help="Path to trained model file (.pth)"
    )
    vis_parser.add_argument(
        "--data", 
        required=True, 
        help="Path to JSON file with climbing data"
    )
    vis_parser.add_argument(
        "--climb-id", 
        help="Specific climb ID to generate for (random if not specified)"
    )
    vis_parser.add_argument(
        "--count", 
        type=int, 
        default=3, 
        help="Number of sequences to generate and visualize"
    )
    vis_parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.8, 
        help="Sampling temperature (higher = more random)"
    )
    vis_parser.add_argument(
        "--output-dir", 
        default="visualizations", 
        help="Directory to save visualizations"
    )
    
    return parser

def train_model(args):
    print(f"Starting training with data from: {args.data}")
    print(f"Training parameters: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    generator = ClimbGenerator(args.data)
    
    #train LSTM
    train_losses, val_losses = generator.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_path=args.output
    )
    
    print(f"Training complete. Model saved to {args.output}")
    print(f"Training loss history saved to training_history.png")
    
    return generator

def generate_sequences(args):
    print(f"Generating {args.count} sequences using model {args.model}")
    
    #make output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    #Initialize generator and load model
    generator = ClimbGenerator(args.data)
    if not generator.load_model(args.model):
        print("Failed to load model. Exiting.")
        return
    
    #gen sequences
    sequences = generator.generate(
        climb_id=args.climb_id,
        num_sequences=args.count,
        temperature=args.temperature,
        max_length=args.max_length
    )
    
    if args.export_json:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(args.output_dir, f"sequences_{timestamp}.json")
        generator.export_sequences(sequences, json_path)
    
    return sequences

def visualize_sequences(args):
    print(f"Visualizing {args.count} sequences using model {args.model}")
    
    #make output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    #load model
    generator = ClimbGenerator(args.data)
    if not generator.load_model(args.model):
        print("Failed to load model. Exiting.")
        return
    
    #gen
    sequences = generator.generate(
        climb_id=args.climb_id,
        num_sequences=args.count,
        temperature=args.temperature
    )
    
    #make valid sewuends 
    valid_sequences = [seq for seq in sequences if seq['is_valid']]
    
    if not valid_sequences:
        print("No valid sequences generated. Try increasing the count or using a different climb ID.")
        return
    
    print(f"Generated {len(valid_sequences)} valid sequences")
    
    #make a viz for each sequence
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, seq in enumerate(valid_sequences):
        filename = os.path.join(args.output_dir, f"sequence_{timestamp}_{i+1}.png")
        generator.visualize_sequence(
            sequence=seq['sequence'],
            climb_id=seq['climb_id'],
            save_path=filename
        )
        print(f"Visualization {i+1} saved to {filename}")
    
    return valid_sequences

def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    print(f"Running in {args.mode} mode")
    
    try:
        if args.mode == "train":
            train_model(args)
        elif args.mode == "generate":
            generate_sequences(args)
        elif args.mode == "visualize":
            visualize_sequences(args)
        else:
            print(f"Unknown mode: {args.mode}")
            parser.print_help()
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()