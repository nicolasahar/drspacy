#!/usr/bin/env python
# coding: utf8
"""Extracts the entities from the Medical Concept Normalization (MCN) corpus and formats them for use with SpaCy's
NER model.

Call `python mcn_to_spacy_entities.py --help` for detailed usage instructions.
"""

import pickle
from pathlib import Path

import fire

# TODO (John): For now, all entity types are hardcoded. In the future we can extract these from other n2c2 datasets
ENTITY_LABEL = "CUI"


def main(input_dir: str, output_dir: str) -> None:
    """Extracts the entities from the MCN corpus and formats them for use with SpaCy's NER model.

    For a copy of the Medical Concept Normalization (MCN) at `input_dir`, extracts the entities from the train
    and test partitions and formats them for use with SpaCy's NER model. The formatted entities are pickled and
    saved to `output_dir/"spacy_formatted_data.pickle"`. This pickle contains a dictionary keyed by partition
    (`"train"`, "`test`").

    Args:
        input_dir (str): Path to top-level directory of the Medical Concept Normalization (MCN) corpus.
        output_dir (str): Path to save the extracted entities formatted for use with SpaCy. The entities
            are pickled and save to a file: `"spacy_formatted_data.pickle"`. This directory will be created if it
            does not already exist.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spacy_formatted_data = {"train": [], "test": []}

    for partition in spacy_formatted_data:
        note_dir = input_dir / partition / f"{partition}_note"
        if partition == "train":
            norm_dir = input_dir / partition / f"{partition}_norm"
        else:
            # TODO (John): Is this the name of the directory in the official download?
            norm_dir = input_dir / partition / f"{partition}_norm_cui_replaced_with_unk"

        for note_filepath in note_dir.iterdir():
            if not note_filepath.name.endswith(".txt"):
                continue

            norm_filepath = norm_dir / f"{note_filepath.stem}.norm"

            with open(note_filepath, "r") as f:
                spacy_formatted_data[partition].append((f.read(), {"entities": []}))
            with open(norm_filepath, "r") as f:
                for line in f:
                    start, end = tuple(map(int, line.strip().split("||")[-2:]))
                    spacy_formatted_data[partition][-1][-1]["entities"].append((start, end, ENTITY_LABEL))

    with open(output_dir / "spacy_formatted_data.pickle", "wb") as f:
        pickle.dump(spacy_formatted_data, f)


if __name__ == "__main__":
    fire.Fire(main)
