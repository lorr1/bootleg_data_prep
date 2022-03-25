"""Convets chinese bootleg data."""
import opencc
import argparse
from pathlib import Path
import ujson
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='output', help='Directory for data.')
    parser.add_argument('--conversion', type=str, default='t2s', help='Converstion (see https://github.com/BYVoid/OpenCC).')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    converter = opencc.OpenCC(args.conversion)
    assert data_dir.exists(), f"{data_dir} does not exist"

    for file in ["dev.jsonl", "test.jsonl", "train.jsonl"]:
        file = Path(file)
        save_file = file.name.replace(".jsonl", "_traditional.jsonl")
        total_l = sum([1 for _ in open(data_dir / file)])
        print(f"Reading {data_dir/file}.")
        with open(data_dir / file) as in_f, open(data_dir / save_file, "w") as out_f:
            simple_lines = []
            for line in tqdm(in_f, total=total_l):
                out_f.write(line.strip() + "\n")
                line = ujson.loads(line)
                new_aliases = [converter.convert(al) for al in line["aliases"]]
                new_sent = converter.convert(line["sentence"])
                assert len(new_sent) == len(line["sentence"]), f"{line}"
                assert all([len(a[0]) == len(a[1]) for a in zip(new_aliases, line["aliases"])])
                line["aliases"] = new_aliases
                line["sentence"] = new_sent
                simple_lines.append(line)
        print(f"Saved original data at {data_dir / save_file}.")
        with open(data_dir / file, "w") as out_f:
            for line in simple_lines:
                out_f.write(ujson.dumps(line, ensure_ascii=False) + "\n")
        print(f"Saved converted simple data at {data_dir / file}.")

if __name__ == '__main__':
    main()