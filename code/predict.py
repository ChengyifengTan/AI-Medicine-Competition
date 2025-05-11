import argparse
from src.predict import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Inference for Cervical MRI Multi-Task Model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to the configuration file'
    )
    args = parser.parse_args()
    run(args.config)
