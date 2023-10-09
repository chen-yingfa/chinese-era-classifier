from tap import Tap
from typing import Union


class Args(Tap):
    num_epochs: int = 4
    batch_size: int = 16  # ~16 for 12GB GPU
    lr: float = 3e-5
    log_interval: int = 40
    pretrained_name: str = "hsc748NLP/GujiRoBERTa_jian_fan"
    num_examples: Union[int, None] = 500000  # Set to None to use all examples

    mode: str = 'train-test'
    output_dir: str = 'result'
    data_dir: str = '../data/souyun'
