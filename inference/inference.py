import time
import sys
import loader
import logging
import json
from pathlib import Path

import yaml
import numpy as np
from transformers import AutoModelForTokenClassification, pipeline

from flattentei.extract_parts import get_units


def run(cfg):
    if cfg.get("ten_fold", False):
        run_ten_fold(cfg)
    else:
        run_standard(cfg)


def run_ten_fold(cfg):
    logging.info("started 10 fold test set inference")
    with open(cfg["input"].get("path_fold_filelist")) as f:
        fold_files = json.load(f)
    for idx_fold, fold_doc_files in fold_files.items():
        # if idx_fold == str(0):
        #    logging.error(f"Skip fold {idx_fold}")
        #    continue
        doc_iter = doc_iterator(cfg, fold_doc_files)
        for step in cfg["models"]["steps"]:
            step["fold"] = idx_fold
        logging.info("Load pipelines")
        start = time.time()
        pipelines = get_pipes(cfg)
        end = time.time()
        logging.info(f"Pipelines loaded in {end-start:.2f} seconds")
        infer_and_save(doc_iter, pipelines, cfg)


def run_standard(cfg):
    batch_size = cfg["models"]["batch_size"]
    doc_iter = doc_iterator(cfg)
    logging.info("Load pipelines")
    start = time.time()
    pipelines = get_pipes(cfg)
    end = time.time()
    infer_and_save(doc_iter, pipelines, cfg)


def doc_iterator(cfg, fold_files=None):
    if cfg["input"]["onefile"] == False:
        doc_iter = loader.load(
            cfg["input"]["path"], limit=cfg["input"].get("limit"), allow_list=fold_files
        )
    else:
        doc_iter = loader.load_one_file(cfg["input"]["path"], cfg["input"].get("limit"))
    return doc_iter


def infer_and_save(doc_iter, pipelines, cfg):
    batch_size = cfg["models"]["batch_size"]
    unit_type = cfg["input"]["unit"]
    unit_type = "Doc" if unit_type is None else unit_type
    all_predictions = []
    while True:
        docs = get_batch(doc_iter, batch_size)
        if not docs:
            break
        # docs = [next(doc_iter) for _ in range(batch_size)]
        logging.info(f"# documents: {len(docs)}")
        if cfg["input"]["path_doc_meta"]:
            loader.replace_ids(
                docs, cfg["input"]["path_doc_meta"], cfg["input"]["meta_id"]
            )
        # get all Paragraphs or sentences
        if cfg["input"]["unit"] is not None:
            texts = []
            for doc in docs:
                units = get_units(unit_type, doc)
                for unit in units:
                    unit["doc_id"] = doc["id"]
                    texts.append(unit)
        else:
            texts = docs
        logging.warning(f"# text units: {len(texts)}")
        logging.info("Predict entities")
        start = time.time()
        predictions = predict_ents(texts, pipelines)
        end = time.time()
        elapsed_time = end - start
        logging.info(
            f"Entities predicted in {elapsed_time:.2f} seconds ({elapsed_time / len(texts):.2f} seconds per {unit_type}, {elapsed_time / len(docs):.2f} seconds per doc)"
        )
        if cfg["output"]["return_units"] == False:
            predictions = join_predictions(docs, predictions, "prediction")
        if cfg["output"]["onefile"] == False:
            loader.dump(predictions, cfg["output"]["path"])
        else:
            all_predictions.extend(predictions)
    if cfg["output"]["onefile"] == True:
        with open(cfg["output"]["path"], "w") as f:
            json.dump(all_predictions, f)


def get_pipes(cfg):
    device = cfg["device"]
    date = cfg["date"]
    base_path = cfg["models"]["path_model_base"]
    pipes = []
    for model_cfg in cfg["models"]["steps"]:
        pipes.append(get_pipe(model_cfg, date, base_path, device))
    return pipes


def get_pipe(model_cfg, date, base_path, device):
    name = model_cfg["name"]
    unit = model_cfg["unit"]
    fold = model_cfg["fold"]
    tokenizer = (model_cfg["tokenizer"], {"model_max_length": 512})
    model_path = get_model_path(base_path, name, date, unit, fold)
    model_path = get_last_model(model_path)
    if model_cfg["model_type"] == "token_classification":
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        pipe = pipeline("ner", model=model, tokenizer=tokenizer, device=device)
        return pipe
    else:
        raise Exception(f"{model_cfg['model_type']} not supported")


def predict_ents(docs, pipelines, prediction_field="prediction"):
    texts = [d["text"] for d in docs]
    pipe_annotations = []
    for pipe in pipelines:
        prediction_raw = pipe(texts, aggregation_strategy="none", ignore_labels=[])
        annotations = predictions_to_annotation_tuple(prediction_raw)
        pipe_annotations.append(annotations)
    # combine all annotations
    annotations = []
    for annos_separate in zip(*pipe_annotations):
        doc_annos = []
        for annos in annos_separate:
            doc_annos.extend(annos)
        annotations.append(doc_annos)
    for doc, annos in zip(docs, annotations):
        enrich_doc(doc, annos, prediction_field)
    return docs


def enrich_doc(doc, annotations, annotation_key):
    doc_annotations = []
    for anno_tuple in annotations:
        anno = anno_tuple_to_dict(anno_tuple)
        enrich_anno(anno, doc["text"])
        doc_annotations.append(anno)
    old_annos = doc.get(annotation_key, [])
    doc_annotations = old_annos + doc_annotations
    doc[annotation_key] = doc_annotations


def predictions_to_annotation_tuple(predictions):
    predictions = [prediction_to_annotation_tuple(p) for p in predictions]
    return predictions


def prediction_to_annotation_tuple(prediction):
    # using bio tag scheme
    spans = []
    span_start = -1
    span_end = -1
    scores = []
    seq_label = []
    seq_token = []
    for anno in prediction + [None]:
        if anno is not None:
            anno["score"] = float(anno["score"])  # float32 is not json serializable
        # POTENTIAL ENTITY ENDING:
        #  token is the last one (None)
        if anno is None:
            if span_start != -1:  # there is an open annotation left
                # add new span end
                spans.append(
                    (span_start, span_end, span_label, scores, seq_label, seq_token)
                )
            break  # last annotation is added.
        # in word token. Add when there is an open annotation left
        elif anno["word"].startswith("##") and span_start != -1:
            # do not add scores for in word token ???
            span_end = anno["end"]
            seq_label.append(anno["entity"])
            seq_token.append(anno["word"])
        # Potential entity ending
        #  A new entity is beginning ("B-") OR
        #  A non entity word begins: ("O")
        elif anno["entity"].startswith("B-") or anno["entity"].startswith("O"):
            if span_start != -1:  # there is a previous ent which need to be added.
                ## not used:
                """
                # last entity and this entity has no space in between "("
                # look for ending in last entity
                # enlarge the last entity and add old label
                if spans and spans[-1][1] == span_start:
                    score_to_add = [anno["score"]]
                    seq_label_to_add = [anno["entity"]]
                    token_to_add = [anno["word"]]
                    last_begin = spans[-1][0]
                    new_scores = spans[-1][3] + score_to_add
                    new_seq_label = spans[-1][4] + seq_label_to_add
                    new_seq_token = spans[-1][5] + token_to_add
                    spans.pop()
                    spans.append((last_begin, span_end, span_label, new_scores, new_seq_token, new_seq_token))
                # add old entity to the span stack
                else:
                    spans.append((span_start, span_end, span_label, scores, seq_label, seq_token))
                """
                spans.append(
                    (span_start, span_end, span_label, scores, seq_label, seq_token)
                )
                span_start, span_end, scores, seq_label, seq_token = -1, -1, [], [], []
        # Now decide to create new entity
        if anno["entity"].startswith("B-"):
            span_label = anno["entity"][2:]  # exclude "B-"
            span_start = anno["start"]
            span_end = anno["end"]
            scores.append(anno["score"])
            seq_label.append(anno["entity"])
            seq_token.append(anno["word"])
        elif anno["entity"].startswith("I-"):
            span_end = anno["end"]
            scores.append(anno["score"])
            seq_label.append(anno["entity"])
            seq_token.append(anno["word"])
        else:
            pass
    spans = tuple(spans)
    return spans


def anno_tuple_to_dict(anno_tuple):
    begin, end, label, scores, seq_label, seq_token = anno_tuple
    return dict(
        label=label,
        begin=begin,
        end=end,
        seq_label=seq_label,
        seq_token=seq_token,
        seq_scores=scores,
    )


def enrich_anno(anno, text):
    score = float(np.mean(anno["seq_scores"]))
    anno_text = text[anno["begin"] : anno["end"]]
    anno["text"] = anno_text
    anno["score"] = score


def join_predictions(docs, predictions, key):
    doc_dict = {d["id"]: idx for idx, d in enumerate(docs)}
    for para in predictions:
        doc_idx = doc_dict[para["doc_id"]]
        doc = docs[doc_idx]
        doc_preds = doc["annotations"].get("ScholarlyEntity", [])
        for anno in para[key]:
            anno["begin"] += para["begin"]
            anno["end"] += para["begin"]
            anno["type"] = "ScholarlyEntity"
            doc_preds.append(anno)
        doc["annotations"]["ScholarlyEntity"] = doc_preds
    return docs


# smaller helper functions


def get_model_path(model_path, model_name, date, unit, fold):
    # Example: 2023-03-24/Sentence/0/scibert_scivocab_cased/"
    ner_model_path = f"{model_path}/{date}/{unit}/{model_name}/{fold}"
    return ner_model_path


def get_last_model(model_path):
    sub_paths = list(Path(model_path).iterdir())
    sub_path_with_id = [(int(str(p).split("-")[-1]), p) for p in sub_paths]
    sub_path_with_id.sort(reverse=True)
    last_ner_model = str(sub_path_with_id[0][1])
    return last_ner_model


def get_batch(doc_iter, batch_size):
    docs = []
    # collect batch docs from iterator
    for _ in range(batch_size):
        try:
            docs.append(next(doc_iter))
        except StopIteration:
            break
    return docs


if __name__ == "__main__":
    # logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    fn_config = sys.argv[1]
    device = int(sys.argv[2])
    # fn_config = "inference/configs/inference-v0.0.yaml"
    # fn_config = "inference/configs/heddes-inference-v0.0.yaml"
    # fn_config = "inference/configs/inference_berd_test_fulltexts-v0.0.yaml"
    with open(fn_config) as f:
        config = yaml.safe_load(f)
    config["device"] = device
    run(config)
