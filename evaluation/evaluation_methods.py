from bisect import bisect_left, bisect_right

import pandas as pd

def calc_scores(gold, prediction):
    scores = prec_recall(gold, prediction, "precision")\
                .join(prec_recall(gold, prediction, "recall"))
    add_f1(scores)
    return scores

def prec_recall(gold, prediction, score_name):
    if score_name == "recall":
        prediction, gold = gold, prediction
    elif score_name != "precision":
        raise Exception("no valid score")
    scores_interim = remap_annos_best_match_many_documents(gold,
                                                prediction)
    scores_interim = pd.DataFrame(scores_interim)
    label_p = scores_interim.label.value_counts()
    p = label_p.sum()
    scores_interim["partial_tp"] = scores_interim.label == scores_interim.label_mapped
    scores_interim["exact_tp"] = (scores_interim.label == scores_interim.label_mapped)\
                                 & (scores_interim.match_type == "exact_match")
    partial_score = (scores_interim.groupby("label").partial_tp.sum() /\
                     label_p).rename(f"partial_{score_name}")
    partial_score.loc["all"] = scores_interim.partial_tp.sum() / p
    exact_score = (scores_interim.groupby("label").exact_tp.sum() / label_p)\
                        .rename(f"exact_{score_name}")
    exact_score.loc["all"] = scores_interim.exact_tp.sum() / p
    score = partial_score.to_frame().join(exact_score)
    return score


def remap_annos_best_match_df(doc_annos, exact=False):
    gold = doc_annos.gold
    pred = doc_annos.pred
    result = remap_annos_best_match_many_documents(gold, pred, exact=False)
    return result


def remap_annos_best_match_many_documents(gold, pred, exact=False):
    """
    each annotation (gold and prediction) need to contain: "begin", "end"
    The documents will be grouped by "id" before matching.
    Each "id" represents one document.
    """
    document_annos = get_document_dict(gold, pred)
    all_annos = []
    for doc, annos in document_annos.items():
        #print(doc)
        #print(len(annos["gold"]))
        #print(len(annos["prediction"]))
        remaped = remap_annos_best_match(annos["gold"], annos["prediction"], exact=exact)
        all_annos.extend(remaped)
    return all_annos
    
def get_document_dict(gold, pred):
    idents = {g["doc_id"] for g in gold}
    idents |= {p["doc_id"] for p in pred}
    print("n_documents:", len(idents))
    by_ident = {ident: dict(gold=[], prediction=[])
                for ident in idents}
    for g in gold:
        by_ident[g["doc_id"]]["gold"].append(g)
    for p in pred:
        by_ident[p["doc_id"]]["prediction"].append(p)
    return by_ident
    
    
def remap_annos_best_match(gold, pred, exact=False):
    pred_remapped = []
    for idx, g in enumerate(gold):
        g["index"] = idx
    gold_begin = sorted(gold, key=lambda x:x["begin"])
    gold_end = sorted(gold, key=lambda x:x["end"])
    for p in pred:
        overlaps = get_overlapping(p, gold_begin, gold_end)
        overlaps = [gold[idx] for idx in overlaps]
        # find perfect match
        match_type = "no_match"
        matched = []
        # find perfect match
        if not matched:
            overlaps_selected = [o for o in overlaps if o["label"] == p["label"]
                                 and (o["begin"] == p["begin"] and o["end"] == p["end"])
                                ]
            if overlaps_selected:
                matched = overlaps_selected
                match_type = "exact_match"
        if not matched and not exact:
            # find same label but only overlapping span
            overlaps_selected = [o for o in overlaps if o["label"] == p["label"]
                        and (o["begin"] != p["begin"] or o["end"] != p["end"])
                       ]
            if overlaps_selected:
                matched = overlaps_selected
                match_type = "partly_match"
        if not matched:
            # find different label but same span
            overlaps_selected = [o for o in overlaps if o["label"] != p["label"]
                        and (o["begin"] == p["begin"] and o["end"] == p["end"])
                       ]
            if overlaps_selected:
                matched = overlaps_selected
                match_type = "exact_span_match"
        if not matched and not exact:
            # find different label and different overlapping span
            overlaps_selected = [o for o in overlaps if o["label"] != p["label"]
                        and (o["begin"] != p["begin"] or o["end"] != p["end"])
                       ]
            if overlaps_selected:
                matched = overlaps_selected
                match_type = "partly_span_match"
        match = {} if not matched else matched[0]
        match = {k:v for k, v in match.items()}
        labels = sorted(list(set([a["label"] for a in matched])))
        labels = " ".join(labels)
        p = {k:v for k, v in p.items()}
        p["text_mapped"] = match.get("text")
        p["begin_mapped"] = match.get("begin")
        p["end_mapped"] = match.get("end")
        p["label_mapped"] = match.get("label", "not_found")
        p["labels_mapped"] = labels if labels else "not_found"
        p["n_matched"] = len(matched)
        p["match_type"] = match_type
        pred_remapped.append(p)
    return pred_remapped

def get_overlapping(anno, annos_begin, annos_end):
    # all annos with smaller end as begining of query anno
    anno_idx_end_max = bisect_right(annos_end, anno["begin"], key=lambda x:x["end"])
    # all annos with smaller beginning as ending of query anno
    anno_idx_begin_min = bisect_left(annos_begin, anno["end"], key=lambda x:x["begin"])
    #anno_idx_begin_min, anno_idx_end_max

    matches = {a["index"] for a in annos_end[anno_idx_end_max:]} &\
              {a["index"] for a in annos_begin[:anno_idx_begin_min]}
    return matches

def add_f1(scores):
    for t in ["partial", "exact"]:
        scores[f"{t}_f1"] = 2 * (scores[f"{t}_precision"] * scores[f"{t}_recall"]) /\
                                (scores[f"{t}_precision"] + scores[f"{t}_recall"])



def print_annos(annos):
    for a in annos:
        print("\t", end="")
        print_anno(a)

def print_anno(anno):
    print(f'{anno["begin"]} {anno["end"]} {anno["label"]}')
