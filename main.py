import argparse
import sys
from src.collector import collect_logs
from src.preprocessor import preprocess_logs
from src.trainer import train_model
from src.prefetcher import do_prefetch
from src.evaluator import evaluate_system

def main():
    parser = argparse.ArgumentParser(description="AI Based File Prefetcher System")
    parser.add_argument(
        "stage", 
        choices=['collect', 'process', 'train', 'prefetch', 'evaluate'],
        help="Execute a specific stage of the machine learning pipeline."
    )
                        
    args = parser.parse_args()
    
    if args.stage == 'collect':
        collect_logs()
    elif args.stage == 'process':
        preprocess_logs()
    elif args.stage == 'train':
        train_model()
    elif args.stage == 'prefetch':
        do_prefetch()
    elif args.stage == 'evaluate':
        evaluate_system()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
