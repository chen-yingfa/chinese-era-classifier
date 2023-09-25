from tap import Tap


class Args(Tap):
    num_epochs: int = 4
    batch_size: int = 64
    lr: float = 3e-5
    log_interval: int = 40
    pretrained_name: str = "hsc748NLP/GujiRoBERTa_jian_fan"
    num_examples: int | None = None

    mode: str = 'train-test'
    output_dir: str = 'result'
    data_dir: str = '../data/souyun'
