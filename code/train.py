import yaml
from src.trainer import train_loop

def main():
    cfg = yaml.safe_load(open('configs/config.yaml', 'r'))
    train_loop(cfg)

if __name__ == '__main__':
    main()
