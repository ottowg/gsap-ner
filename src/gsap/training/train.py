from pathlib import Path
import sys
import os
import re
import logging
from functools import partial
import pkg_resources

from datasets import load_dataset, load_from_disk
from transformers import  DataCollatorForTokenClassification,\
                          AutoModelForTokenClassification,\
                          TrainingArguments,\
                          Trainer,\
                          AutoTokenizer
import yaml

from .prepare_dataset import prepare_dataset
from .metrics import compute_metrics
    

def train_from_config(config):
    c = config
    n_folds = c["data"]["n_folds"]
    path_base_data = get_dataset_path(c)
    path_base_model = get_model_path(c)
    print(path_base_data)
    print(path_base_model)
    for fold in range(0, n_folds):
        print(f"Start with fold {fold}")
        
        logging.info("load data")
        path_fold = path_base_data / Path(str(fold))
        dataset_raw = load_from_disk(path_fold)
        dataset_tokenized, label_names = prepare_dataset(dataset_raw,
                                                         c["model"]["base_model"],
                                                         c["data"]["tagset"])
        print(label_names)
        
        # load model
        id2label = {i: label for i, label in enumerate(label_names)}
        label2id = {v: k for k, v in id2label.items()}
        model = AutoModelForTokenClassification.from_pretrained(
            c["model"]["base_model"],
            ignore_mismatched_sizes=True,
            id2label=id2label,
            label2id=label2id,
            )
        # prepare collator for padding batches
        tokenizer = AutoTokenizer.from_pretrained(c["model"]["base_model"])
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        logging.info(f"training instances: {dataset_tokenized['train']}") 
        logging.info("train the model")
        path_model_fold = path_base_model / Path(str(fold))
        #freeze_layer(model, up_to=5)
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                path_model_fold,
                **c["model"]["training_arguments"]
            ),
            train_dataset=dataset_tokenized["train"],
            eval_dataset=dataset_tokenized["validation"],
            data_collator=data_collator,
            compute_metrics=partial(compute_metrics, label_names=label_names),
            tokenizer=tokenizer,
        )
        trainer.train()

def get_dataset_path(c):
    path_base = Path(c["data"]["path"])
    if not path_base.is_absolute():
        path_base = Path(os.getcwd()) / path_base
    if "date" in c:
        path_base /= Path(c["date"])
    path_base /= Path(c["data"]["unit"])
    return path_base

def get_model_path(c):
    path_base = Path(c["model"].get("path"))
    if not path_base.is_absolute():
        path_base = Path(os.getcwd()) / path_base
    model_name = f'{Path(c["model"]["base_model"]).name}-{c["data"]["tagset"]}-'
    model_name += c["model"].get("nickname", "")
    model_path = Path(path_base) /\
                 Path(c["data"]["unit"]) /\
                 Path(model_name)
    return model_path

def freeze_layer(model, up_to):
    for weight_id, param in model.named_parameters():
        if weight_id.startswith("bert.embedding"):
            param.requires_grad = False
        layer_num = re.findall("^bert\.encoder\.layer.(?P<num>[0-9]+).*", weight_id)
        layer_num = layer_num[0] if layer_num else None
        if layer_num and int(layer_num) <= up_to:
            param.requires_grad = False

def main():
    logging.getLogger().setLevel(logging.INFO)
    fn_config = sys.argv[1]
    if len(sys.argv) >= 3:
        device = sys.argv[2]
        os.environ["CUDA_VISIBLE_DEVICES"] = device # e.g. "0,1,2"
    with open(fn_config) as f:
        config = yaml.safe_load(f)
        train_from_config(config)

        
if __name__ == "__main__":
    main()
