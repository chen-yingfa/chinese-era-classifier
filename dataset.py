from pathlib import Path
from typing import Optional, Union
import random

from tqdm import tqdm
import torch
from torch import LongTensor
from torch.utils.data import Dataset

from utils import load_json


def load_souyun_examples(examples_path: Path, cnt: Optional[int] = None) -> list[dict]:
    print(f"Loading examples from {examples_path}")
    raw_examples = load_json(examples_path)[:cnt]
    dynasty_name_map = {
        "秦": "秦汉",
        "汉": "秦汉",
        "辽": "宋",
        "金": "宋",
        "隋": "隋唐",
        "唐": "隋唐",
        "魏晋": "魏晋南北朝",
        "南北朝": "魏晋南北朝",
    }
    dynasty_name_map = {
        "先秦": "上古汉语",
        "秦": "上古汉语",
        "汉": "上古汉语",
        "辽": "中古汉语",
        "金": "中古汉语",
        "隋": "中古汉语",
        "唐": "中古汉语",
        "魏晋": "中古汉语",
        "南北朝": "中古汉语",
        "宋": "中古汉语",
        "元": "近代汉语",
        "明": "近代汉语",
        "清": "近代汉语",
        "近代": "近代汉语",
        "当代": "近代汉语",
        "近现代": "近代汉语",
    }
    labels = ["上古汉语", "中古汉语", "近代汉语"]

    # We try to have the same number of examples for each label
    label_to_examples = {k: [] for k in labels}
    for i, eg in enumerate(tqdm(raw_examples)):
        if all(len(label_to_examples[k]) >= cnt for k in labels):
            break
        if eg["dynasty_name"] in dynasty_name_map:
            label = dynasty_name_map[eg["dynasty_name"]]
        else:
            label = eg["dynasty_name"]
        assert label in labels, label
        if len(label_to_examples[label]) >= cnt:
            continue
        for sent in eg["content"]:
            label_to_examples[label].append(
                {
                    "text": sent,
                    "label": label,
                }
            )
        label_to_examples[label].append(
            {
                "text": eg["title"],
                "label": label,
            }
        )
    examples = sum(label_to_examples.values(), [])
    examples = random.sample(examples, cnt)
    return examples


def create_features(
    examples: list[dict], tokenizer, label_map: dict[str, int]
) -> dict[str, LongTensor]:
    texts = [eg["text"] for eg in examples]
    labels = [label_map[eg["label"]] for eg in examples]
    print("Tokenizing texts...")
    tokens: dict[str, LongTensor] = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "label": LongTensor(labels),
    }


def load_features(
    features_path: Path,
    examples_path: Path,
    tokenizer,
    label_map: dict[str, int],
    num_examples: Optional[int] = None,
) -> dict[str, LongTensor]:
    if features_path.exists():
        print(f"Loading features from {features_path}")
        return torch.load(features_path)
    else:
        print(f"Features not found at {features_path}, creating from examples")
        examples = load_souyun_examples(examples_path, cnt=num_examples)
        features = create_features(examples, tokenizer, label_map)
        features_path.parent.mkdir(exist_ok=True, parents=True)
        print(f"Saving features to {features_path}")
        torch.save(features, features_path)
        return features


class EraDataset(Dataset):
    # class_names = ["先秦", "秦汉", "魏晋南北朝", "隋唐", "宋", "元", "明", "清", "近现代", "当代"]
    class_names = ["上古汉语", "中古汉语", "近代汉语"]

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer,
        cache_dir: str = ".data_cache",
        num_examples: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.cache_dir = Path(cache_dir)
        self.num_examples = num_examples

        # label_map = {label: i for i, label in enumerate(self.class_names)}
        # features_path = self.cache_dir / f"{self.data_path.stem}.pt"
        self.examples = load_souyun_examples(self.data_path, cnt=num_examples)
        # self.features: dict = load_features(
        #     features_path, self.data_path, tokenizer, label_map, num_examples
        # )

    def __getitem__(self, index) -> dict[str, LongTensor]:
        return self.examples[index]
        eg = {
            key: self.features[key][index]
            for key in ["input_ids", "attention_mask", "label"]
        }
        return eg

    def __len__(self) -> int:
        return len(self.examples)
        # return len(self.features["label"])
