"""Convets chinese bootleg data."""
import opencc
import argparse
from pathlib import Path
import shutil
import ujson
from tqdm.auto import tqdm

from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.symbols.entity_profile import EntityProfile
from bootleg.symbols.type_symbols import TypeSymbols
from bootleg.symbols.kg_symbols import KGSymbols


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='output', help='Directory for data.')
    parser.add_argument('--conversion', type=str, default='t2s',
                        help='Converstion (see https://github.com/BYVoid/OpenCC).')
    args = parser.parse_args()
    return args


def convert_text(data_dir, converter):
    for file in ["dev.jsonl", "test.jsonl", "train.jsonl"]:
        file = Path(file)
        save_file = file.name.replace(".jsonl", "_traditional.jsonl")
        total_l = sum([1 for _ in open(data_dir / file)])
        print(f"Reading {data_dir / file}.")
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
    return


def convert_entity_db(data_dir, converter):
    # Load entity desctiption and add to entity symbols
    print(f"Saving original entity db at {data_dir}/entity_db_traditional")
    if (data_dir / "entity_db_traditional").exists():
        shutil.rmtree(data_dir / "entity_db_traditional")
    shutil.copytree(data_dir / "entity_db", data_dir / "entity_db_traditional")
    ep = EntityProfile.load_from_cache(data_dir / "entity_db")

    print("Making entity symbols")
    alias2qids = {converter.convert(al): v for al, v in ep._entity_symbols.get_alias2qids_dict().items()}
    qid2title = {q: converter.convert(t) for q, t in ep._entity_symbols.get_qid2title_dict().items()}
    qid2desc = {q: converter.convert(d) for q, d in ep._entity_symbols._qid2desc.items()}
    entity_symbols = EntitySymbols(
        alias2qids=alias2qids,
        qid2title=qid2title,
        qid2desc=qid2desc,
        max_candidates=ep._entity_symbols.max_candidates
    )
    print("Built entity symbols")

    qid2relations = {q: {converter.convert(k): v for k, v in rels.items()} for q, rels in ep._kg_symbols.get_qid2relations_dict().items()}
    kg_symbols = KGSymbols(
        qid2relations=qid2relations,
        max_connections=ep._kg_symbols.max_connections
    )
    print("Built kg symbols")

    qid2typenames = {q: [converter.convert(t) for t in typs] for q, typs in ep._type_systems["wiki"].get_qid2typename_dict().items()}
    type_symbols = TypeSymbols(qid2typenames=qid2typenames, max_types=ep._type_systems["wiki"].max_types)
    entity_profile = EntityProfile(
        entity_symbols=entity_symbols,
        type_systems={"wiki": type_symbols},
        kg_symbols=kg_symbols
    )
    print("Built type symbols")
    del ep

    entity_profile.save(data_dir / "entity_db")
    print(f"Saved new profile at {data_dir}/entity_db")
    return


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    converter = opencc.OpenCC(args.conversion)
    assert data_dir.exists(), f"{data_dir} does not exist"
    convert_text(data_dir=data_dir, converter=converter)
    convert_entity_db(data_dir=data_dir, converter=converter)


if __name__ == '__main__':
    main()
