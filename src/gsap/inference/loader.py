from pathlib import Path
import json


def load_one_file(file_path, limit=None):
    with open(file_path) as f:
        docs = json.load(f)
        for idx, doc in enumerate(docs):
            if limit is not None and idx >= limit:
                break
            yield doc


def load(doc_path, limit=None, allow_list=None):
    path = Path(doc_path)
    for idx, fn_doc in enumerate(path.iterdir()):
        if limit is not None and idx >= limit:
            break
        if allow_list is None or fn_doc.name in allow_list:
            doc = json.load(fn_doc.open())
            doc["filename"] = fn_doc.name
            doc["id"] = fn_doc.stem
            yield doc


def replace_ids(docs, path_meta, meta_id):
    replace_dict = {}
    with Path(path_meta).open() as f:
        for line in f:
            rec = json.loads(line)
            old_id = Path(rec["filename"]).stem
            replace_dict[old_id] = rec["doi"]
    for doc in docs:
        doc["id"] = replace_dict[doc["id"]]


def dump(docs, doc_path):
    path = Path(doc_path)
    if not path.exists():
        path.mkdir(parents=True)
    for doc in docs:
        fn = path / Path(doc["filename"]).with_suffix(".json")
        with fn.open("w") as f:
            json.dump(doc, f)
