import torch
from torch import nn, Tensor
from pathlib import Path
import time

from transformers import BertForSequenceClassification, BertTokenizerFast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EraDataset
from utils import dump_json, load_json


def load_model(pretrained_name: str, num_classes: int):
    model = BertForSequenceClassification.from_pretrained(
        pretrained_name,
        num_labels=num_classes,
    ).cuda()
    tok = BertTokenizerFast.from_pretrained(pretrained_name)
    return model, tok


def load_data(data_dir: Path, tokenizer) -> tuple[EraDataset, EraDataset, EraDataset]:
    train_dataset = EraDataset(data_dir / "train.json", tokenizer)
    valid_dataset = EraDataset(data_dir / "valid.json", tokenizer)
    test_dataset = EraDataset(data_dir / "test.json", tokenizer)
    return train_dataset, valid_dataset, test_dataset


def evaluate(
    model, output_dir: Path, valid_dataset: EraDataset, batch_size: int = 2
) -> dict:
    model.eval()
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False
    )
    start_time = time.time()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    total_correct = 0
    total = 0
    print("====== Evaluation start ======")
    for batch_idx, batch in enumerate(tqdm(valid_loader)):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = batch["label"].cuda()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits: Tensor = outputs.logits
            loss: Tensor = loss_fn(logits, labels)
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += len(labels)

    print("====== Evaluation end ======")

    time_elapsed = time.time() - start_time
    avg_loss = total_loss / len(valid_loader)
    avg_acc = total_correct / total
    scores = {
        "loss": avg_loss,
        "acc": avg_acc,
        "time": time_elapsed,
    }
    print(f"Valid loss: {avg_loss:.4f}")
    print(f"Valid acc: {avg_acc:.4f}")
    print(f"Time elapsed: {time_elapsed:.2f}s")
    output_dir.mkdir(exist_ok=True, parents=True)
    dump_json(scores, output_dir / "scores.json")
    return scores


def train(
    model: nn.Module,
    output_dir: Path,
    train_dataset: EraDataset,
    valid_dataset: EraDataset,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 3e-5,
    log_interval: int = 1,
):
    print("train")
    output_dir.mkdir(exist_ok=True, parents=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    loss_fn = nn.CrossEntropyLoss()

    start_time = time.time()
    global_step = 0
    cur_ep = 0
    print("====== Start training ======")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"# of training examples: {len(train_dataset)}")
    print(f"# of validation examples: {len(valid_dataset)}")
    print(f"# batches: {len(train_loader)}")

    dev_results = []
    train_results = []

    while cur_ep < epochs:
        print(f"Epoch {cur_ep + 1}/{epochs}")
        epoch_loss = 0
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            # Forward
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["label"].cuda()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits: Tensor = outputs.logits
            loss: Tensor = loss_fn(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log
            epoch_loss += loss.item()
            global_step += 1
            if global_step % log_interval == 0:
                print(
                    dict(
                        ep=cur_ep,
                        step=batch_idx + 1,
                        loss=round(loss.item(), 6),
                        time=round(time.time() - start_time),
                        lr=round(optimizer.param_groups[0]["lr"], 6),
                    )
                )
        lr_scheduler.step()
        epoch_loss /= len(train_loader)
        ep_time = time.time() - start_time
        print(f"Epoch time: {ep_time:.2f}s")
        print(f"Train loss: {epoch_loss:.4f}")
        train_results.append(
            dict(
                epoch=cur_ep,
                loss=epoch_loss,
                time=ep_time,
            )
        )

        # Validation
        ckpt_dir = output_dir / f"ckpt-{cur_ep}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model, ckpt_dir / "model.pt")
        dev_result = evaluate(model, ckpt_dir, valid_dataset, batch_size=batch_size)
        dev_result["epoch"] = cur_ep
        dump_json(dev_result, ckpt_dir / "dev_result.json")
        valid_loss = dev_result["loss"]
        valid_acc = dev_result["acc"]
        print(f"Valid loss: {valid_loss:.4f}")
        print(f"Valid acc: {valid_acc:.4f}")
        print("====== End of epoch ======")
        cur_ep += 1

        dev_results.append(dev_result)

    print("====== End of training ======")
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")
    dump_json(dev_results, output_dir / "dev_results.json")
    dump_json(train_results, output_dir / "train_results.json")


def load_best_ckpt(output_dir: Path) -> nn.Module:
    best_loss = float("inf")
    best_ckpt = None
    for ckpt_dir in sorted(output_dir.glob("ckpt-*")):
        dev_result = load_json(ckpt_dir / "dev_result.json")
        valid_loss = dev_result["loss"]
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_ckpt = ckpt_dir
    assert best_ckpt is not None
    print(f"Loading best checkpoint from {best_ckpt}")
    return torch.load(best_ckpt / "model.pt")


def main():
    mode = "train-test"
    pretrained_name = "hsc748NLP/GujiRoBERTa_jian_fan"
    output_dir = Path("result", pretrained_name)
    model, tok = load_model(pretrained_name, len(EraDataset.class_names))

    data_dir = Path("../data/souyun")
    train_data, dev_data, test_data = load_data(data_dir, tok)

    if "train" in mode:
        train(model, output_dir, train_data, dev_data)
    if "test" in mode:
        test_output_dir = output_dir / "test"
        test_output_dir.mkdir(exist_ok=True, parents=True)
        evaluate(model, test_output_dir, test_data)


if __name__ == "__main__":
    main()
