import json


def iter_json(path):
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            yield json.loads(line)


def load_json(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def dump_json(data, path):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
