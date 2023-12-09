from functools import partial
import random

from transformers import AutoTokenizer

# transform raw dataset to a ready to train on dataset
# incl. tokenization and label mapping


def prepare_dataset(dataset_raw, model_name, tagset):
    """
    tagset is currently "flat_base", "flat_plus", "stacked"
    last one is not working properly and some annotations are deleted
    """
    ner_label = dataset_raw["train"].features[f"{tagset}_label"].feature.names
    print(ner_label)
    label_names = ["O"] + [f"B-{l}" for l in ner_label] + [f"I-{l}" for l in ner_label]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_datasets = dataset_raw.map(
        partial(
            tokenize_annotated,
            tokenizer=tokenizer,
            tagset=tagset,
            num_tags=len(ner_label),
        ),
        batched=True,
        remove_columns=dataset_raw["train"].column_names,
    )

    sample = random.choice(tokenized_datasets["train"])
    print(sample)
    l = sample["labels"]
    t = sample["token"]
    print(t)
    print(l)
    print(len(t), len(l))
    return tokenized_datasets, label_names


def tokenize_annotated(examples, tokenizer, tagset, num_tags=0):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        is_split_into_words=False,
        # return_offesets_mapping=True
    )

    # tokenized_inputs = []
    num_samples = len(examples)

    ner_start = examples[f"{tagset}_start"]
    ner_end = examples[f"{tagset}_end"]
    ner_labels = examples[f"{tagset}_label"]
    features = (ner_start, ner_end, ner_labels)
    new_labels = []
    subword_token = []
    for idx, (ner_start, ner_end, ner_label) in enumerate(zip(*features)):
        token = tokenized_inputs[idx]
        annos = zip(ner_start, ner_end, ner_label)
        new_label, new_token = align_labels_with_tokens(token, annos, num_tags)
        new_labels.append(new_label)
        subword_token.append(new_token)

    tokenized_inputs["labels"] = new_labels
    tokenized_inputs["token"] = subword_token
    return tokenized_inputs


def align_labels_with_tokens(token, annos, num_tags):
    word_ids = token.word_ids
    subword_token = token.tokens
    # print(word_ids)
    all_label = [0 for i in word_ids]
    all_label[0] = -100
    all_label[-1] = -100
    for start, end, label in annos:
        # print(start, end, label)
        start_token = token.char_to_token(start)
        end_token = token.char_to_token(end - 1)
        if start_token is None or end_token is None:
            print("tokenizer missmatch")
            break
        # print(start_token, end_token, start, end)
        ent_tokens = word_ids[start_token : end_token + 1]
        current_token = -1
        for idx, ent_token in enumerate(ent_tokens):
            ent_idx = start_token + idx
            if idx == 0:  # start
                all_label[ent_idx] = label + 1  # "B-DATASET"
            elif ent_token == current_token:
                all_label[ent_idx] = label + 1 + num_tags  # "X-DATASET" # -100
            else:
                all_label[ent_idx] = label + 1 + num_tags  # "I-DATASET"
            current_token = ent_token
    # raise Exception()
    return all_label, subword_token
