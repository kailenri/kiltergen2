from LSTM.climb_model import ClimbGenerator
from LSTM.climb_interface import ClimbInterface
from config import *

def main():
    # Initialize components
    generator = ClimbGenerator(JSON_PATH)
    
    # Train or load model
    train_new = input("Train new model? (y/n): ").lower() == 'y'
    if train_new:
        generator.train()
    else:
        generator.load_model(LSTM_PATH)
    
    # Start interface
    interface = ClimbInterface(generator)
    interface.start_interactive()

if __name__ == "__main__":
    main()