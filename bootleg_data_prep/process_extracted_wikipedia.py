"""
Parses wikipedia data from WikiExtractor into passages and pageids for future steps.
"""
import argparse
import multiprocessing
import re
import sys
import time
import ujson
from bs4 import BeautifulSoup
from urllib.parse import unquote
import html
from pathlib import Path

import ray
import ujson as json
from tqdm.auto import tqdm
from typing import Dict, List, Any, Tuple

import bootleg_data_prep.utils.data_prep_utils as prep_utils
from bootleg_data_prep.language import sent_offset_tokenize, ENSURE_ASCII


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wikiextractor_output",
        type=str,
        default="data/wikiextractor_output",
        help="Where files saved",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory for data to be saved.",
    )
    parser.add_argument(
        "--processes", type=int, default=int(0.25 * multiprocessing.cpu_count())
    )
    args = parser.parse_args(args)
    return args


@ray.remote
class ExtractProcess(object):
    def __init__(self):
        pass

    def sentence_chunk(self, page_text: str, entity_data: Dict[Tuple[int, int], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split page into sentence and separate relevant mention data."""
        sentence_all_data = []
        sent_idx = 0
        last_cur_offset = -1
        for i, [cur_offset, end_offset] in enumerate(sent_offset_tokenize(page_text)):
            cur_offset = cur_offset if last_cur_offset == -1 else last_cur_offset
            sent = page_text[cur_offset:end_offset]
            sent_data = {
                "aliases": [],
                "char_spans": [],
                "titles": [],
                "sentence": sent,
                "doc_sent_idx": sent_idx
            }
            bad_sent = False
            # Bucketize the entity data by the sentences
            for [span_l, span_r], ent_dict in entity_data.items():
                if span_l >= cur_offset and span_l < end_offset:
                    # Indication that the sentence is split in the middle of a name
                    if span_r > end_offset:
                        bad_sent = True
                        break
                    else:
                        new_span = [ent_dict["char_span"][0]-cur_offset, ent_dict["char_span"][1]-cur_offset]
                        assert sent[new_span[0]:new_span[1]] == ent_dict["alias"], f"{sent} {ent_dict}"
                        sent_data["aliases"].append(ent_dict["alias"])
                        sent_data["titles"].append(ent_dict["title"])
                        sent_data["char_spans"].append(new_span)
            if bad_sent:
                last_cur_offset = cur_offset
                continue
            else:
                sent_idx += 1
                last_cur_offset = -1
                sentence_all_data.append(sent_data)
        return sentence_all_data

    def process_mention_tags(self, raw_text: str) -> Tuple[str, Dict[Tuple[int, int], Dict[str, Any]]]:
        """Extract mention information from page.

        This step parses the HTML output and extracts <a></a> tags. The href points to the Wikipedia
         page and the tag text is the alias.
        """
        # Replace multiple new lines with one
        raw_text = re.sub(r'\n+', '\n', raw_text).strip()
        try:
            soup = BeautifulSoup(raw_text, features="html.parser")
            # Find all mentions
            tags = soup.find_all("a")
        except TypeError:
            print("ERROR TYPES...a few of these are okay.")
            tags = []
        end_tag = "/a>"
        # All page text without HTML links
        page_text = ""
        entity_data = {}
        cur_pos = 0
        final_pos = len(raw_text)
        # Keep track of lines for positioning
        raw_text_lines = raw_text.splitlines()
        for tag_idx, tag in enumerate(tags):
            tag_st = tag.sourcepos
            tag_st_ln = tag.sourceline
            for i in range(tag_st_ln-1):
                # +1 for the single newline -> this is why we made sure there was only one
                tag_st += len(raw_text_lines[i]) + 1
            # bs4 ignores the brackets for sourcepos but we need them to extract the outside bracket text
            tag_end = raw_text.find(end_tag, tag_st) + len(end_tag)

            # Add pre-tag text + mention
            # print("Begining text:", raw_text.replace("\n", "*******")[:1000])
            # print("Tag St", tag_st, "End St", tag_end, "St Line", tag.sourceline)
            # print("Adding", raw_text[cur_pos:tag_st] + "******")
            # print("Adding2", tag.text + "******")
            page_text += raw_text[cur_pos:tag_st] + tag.text
            cur_pos = tag_end

            if len(tag.text) > 0 and tag.get("href") is not None:
                # print("Cur PT", page_text.replace("\n", "********"))
                alias = tag.text
                # # is section headers
                title = html.unescape(unquote(tag.get("href")).split("#")[0].strip())
                span = [len(page_text)-len(tag.text), len(page_text)]
                test_alias = page_text[span[0]:span[1]]
                assert tuple(span) not in entity_data
                entity_data[tuple(span)] = {
                    "alias": alias,
                    "title": title,
                    "char_span": span
                }
                if alias != test_alias:
                    print(raw_text)
                    print(page_text)
                    print(f"Al [{alias}] Test Al [{test_alias}] Span {span} Title [{title}]")
                    import pdb; pdb.set_trace()
        page_text += raw_text[cur_pos:final_pos]
        # print(raw_text)
        # print(page_text)
        return page_text, entity_data

    def subprocess(
        self, i: int, total: int, text_outputdir: Path, pageids_outputdir: Path, in_filepath: Path
    ):
        """Build wikipedia data."""
        print(f"Starting {i}/{total}. Reading in {in_filepath}.")

        text_outfile = text_outputdir / f"wiki_{i}.txt"
        pageid_outfile = pageids_outputdir / f"wiki_{i}.txt"
        num_pages = 0
        with open(text_outfile, "w") as out_text, open(pageid_outfile, "w") as out_page, open(in_filepath) as in_f:
            for page in in_f:
                num_pages += 1
                page = ujson.loads(page)
                # Turn into HTML for bs4
                raw_text = html.unescape(page["text"])
                page_text, entity_data = self.process_mention_tags(raw_text)
                sentence_all_data = self.sentence_chunk(page_text, entity_data)
                out_text.write(ujson.dumps({
                    "page_title": page["title"],
                    "aliases": sentence_all_data
                }, ensure_ascii=ENSURE_ASCII) + "\n")
                out_page.write(ujson.dumps({"title": page["title"], "id": page["id"]}, ensure_ascii=ENSURE_ASCII) + "\n")
        return num_pages


def launch_subprocess(
    text_outputdir: Path,
    pageids_outputdir: Path,
    files: List[Path],
    processes: int,
):
    """Launch Wikipedia postprocessing."""
    process_count = min(max(1, processes), len(files))

    # Start worker processes
    ray.init(ignore_reinit_error=True)
    worker_pool = ray.util.ActorPool(
        [ExtractProcess.remote() for _ in range(process_count)]
    )
    arg_list = [
        {
            "i": idx,
            "total": len(files),
            "text_outputdir": text_outputdir,
            "pageids_outputdir": pageids_outputdir,
            "in_filepath": files[idx],
        }
        for idx in range(len(files))
    ]
    # Cast as list to force execution
    num_pages = 0
    for np in tqdm(
        worker_pool.map_unordered(
            lambda act, val: act.subprocess.remote(**val), arg_list
        ), total=len(arg_list)
    ):
        num_pages += np
    print(f"Processed {num_pages} pages")
    ray.shutdown()
    return


def main(args):
    """Run postprocessing."""
    gl_start = time.time()
    args = parse_args(args)
    print(json.dumps(vars(args), indent=4))

    # Final results
    text_outputdir = Path(prep_utils.get_outdir(args.output_dir, "sentences", remove_old=True))
    pageids_outputdir = Path(prep_utils.get_outdir(args.output_dir, "pageids", remove_old=True))

    # All input files
    files = []
    for subdir in Path(args.wikiextractor_output).glob("*"):
        for file in subdir.glob("wiki_*"):
            files.append(file)

    print(
        f"Loaded {len(files)} files from {args.wikiextractor_output}. Launching {args.processes} processes."
    )

    launch_subprocess(
        text_outputdir=text_outputdir,
        pageids_outputdir=pageids_outputdir,
        files=files,
        processes=args.processes,
    )

    print(f"Finished process_extracted_wikipedia in {time.time() - gl_start} seconds.")


if __name__ == "__main__":
    main(sys.argv[1:])