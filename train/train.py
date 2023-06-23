from pathlib import Path
import sys
import os
import re
import logging
from functools import partial

from datasets import load_dataset, load_from_disk
from transformers import  DataCollatorForTokenClassification,\
                          AutoModelForTokenClassification,\
                          TrainingArguments,\
                          Trainer,\
                          AutoTokenizer
import yaml

from prepare_dataset import prepare_dataset
from metrics import compute_metrics

    

def train_from_config(config):
    c = config
    n_folds = c["data"]["n_folds"]
    for fold in range(0, n_folds):
        print(f"Started with fold {fold}")
        # load data
        c_data = c["data"]
        dataset_path = Path(c["data"]["path"]) /\
                       Path(c["data"]["date"]) /\
                       Path(c["data"]["unit"]) /\
                       Path(str(fold))
        dataset_raw = load_from_disk(dataset_path)
        dataset_tokenized, label_names = prepare_dataset(dataset_raw,
                                                         c["model"]["base_model"],
                                                         c["data"]["tagset"])
        print(label_names)
        #raise Exception("WTF") 
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
        # train the model
        model_name = f'{Path(c["model"]["base_model"]).name}-{c["data"]["tagset"]}-'
        model_name += c["model"].get("nickname", "")
        model_path = Path(c["model"]["path"]) /\
                     Path(c["data"]["date"]) /\
                     Path(c["data"]["unit"]) /\
                     Path(model_name) /\
                     Path(str(fold))
        #freeze_layer(model, up_to=5)
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                model_path,
                **c["model"]["training_arguments"]
            ),
            train_dataset=dataset_tokenized["train"],
            eval_dataset=dataset_tokenized["validation"],
            data_collator=data_collator,
            compute_metrics=partial(compute_metrics, label_names=label_names),
            tokenizer=tokenizer,
        )
        trainer.train()

def freeze_layer(model, up_to):
    for weight_id, param in model.named_parameters():
        if weight_id.startswith("bert.embedding"):
            param.requires_grad = False
        layer_num = re.findall("^bert\.encoder\.layer.(?P<num>[0-9]+).*", weight_id)
        layer_num = layer_num[0] if layer_num else None
        if layer_num and int(layer_num) <= up_to:
            param.requires_grad = False
    
        
if __name__ == "__main__":
    #logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    fn_config = sys.argv[1]
    device = sys.argv[2]
    os.environ["CUDA_VISIBLE_DEVICES"] = device # e.g. "0,1,2"
    with open(fn_config) as f:
        config = yaml.safe_load(f)
        train_from_config(config)

# Example config in yaml
example_config = """
    project: gsap_ner
    data:
      path: /home/ottowg/projects/dataset_extraction/ner_huggingface/gsap
      n_folds: 10
      date: 2023-05-24
      unit: Paragraph
      tagset: flat_base # flat_plus
    model:
      nickname: vanila,
      base_model: allenai/scibert_scivocab_cased
      path: /data_ssds/disk07/ottowg/model/gsap/
      training_arguments:
        evaluation_strategy: epoch
        save_strategy: epoch
        load_best_model_at_end: true
        metric_for_best_model: eval_f1
        save_total_limit: 1
        learning_rate: 0.00001
        warmup_ratio: 0.1
        num_train_epochs: 20
        per_device_train_batch_size: 2,
        per_device_eval_batch_size: 2
        lr_scheduler_type: cosine
        gradient_accumulation_steps: 32 #  needed for better memory perf
        weight_decay: 0.01
        push_to_hub: false
        # group_by_length
"""
