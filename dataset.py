from pathlib import Path

from tqdm import tqdm
import torch
from torch import LongTensor
from torch.utils.data import Dataset

from utils import load_json


def load_souyun_examples(examples_path: Path, cnt: int | None = 10000) -> list[dict]:
    print(f"Loading examples from {examples_path}")
    raw_examples = load_json(examples_path)[:cnt]
    examples = []
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
    for i, eg in enumerate(tqdm(raw_examples)):
        if eg["dynasty_name"] in dynasty_name_map:
            label = dynasty_name_map[eg["dynasty_name"]]
        else:
            label = eg["dynasty_name"]
        for sent in eg["content"]:
            examples.append(
                {
                    "text": sent,
                    "label": label,
                }
            )
        examples.append(
            {
                "text": eg["title"],
                "label": label,
            }
        )
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
    features_path: Path, examples_path: Path, tokenizer, label_map: dict[str, int]
) -> dict[str, LongTensor]:
    if features_path.exists():
        print(f"Loading features from {features_path}")
        return torch.load(features_path)
    else:
        print(f"Features not found at {features_path}, creating from examples")
        examples = load_souyun_examples(examples_path)
        features = create_features(examples, tokenizer, label_map)
        features_path.parent.mkdir(exist_ok=True, parents=True)
        print(f"Saving features to {features_path}")
        torch.save(features, features_path)
        return features


class EraDataset(Dataset):
    class_names = ["先秦", "秦汉", "魏晋南北朝", "隋唐", "宋", "元", "明", "清", "近现代", "当代"]

    def __init__(
        self, data_path: Path | str, tokenizer, cache_dir: str = ".data_cache"
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.cache_dir = Path(cache_dir)

        label_map = {label: i for i, label in enumerate(self.class_names)}
        features_path = self.cache_dir / f"{self.data_path.stem}.pt"
        self.features: dict = load_features(
            features_path, self.data_path, tokenizer, label_map
        )

    def __getitem__(self, index) -> dict[str, LongTensor]:
        eg = {
            key: self.features[key][index]
            for key in ["input_ids", "attention_mask", "label"]
        }
        return eg

    def __len__(self) -> int:
        return 10000
        return len(self.features['label'])
