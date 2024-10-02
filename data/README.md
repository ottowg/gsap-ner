# Dataset

We distribute two datasets in the huggingface format.
The data is used for the publication.
It is based on an inception export from 2023-06-21.

 * The data contains annotation for 100 publications.
 * The data delivered in a 10-fold split.
 * The text unit is paragraph.
 * For each split all paragraphs of 80% of the publications are in train, 10 in validation, 10 in test set. 


## Data format
 * Hugggingface dataset is used to genereate the data in the arrow format.
 * To load the data import: `from datasets import load_from_disk`
 * And load the dataset: `load_from_disk("Paragraph/<fold_idx>") (fold_idx: 0-9)
 
### Overview over features of the paragraphs stored in the dataset
```
"features": {
    "id": {
      "dtype": "string",
      "_type": "Value"
    },
    "text": {
      "dtype": "string",
      "_type": "Value"
    },
    "stacked_start": {
      "feature": {
        "dtype": "int64",
        "_type": "Value"
      },
      "_type": "Sequence"
    },
    "stacked_end": {
      "feature": {
        "dtype": "int64",
        "_type": "Value"
      },
      "_type": "Sequence"
    },
    "stacked_label": {
      "feature": {
        "names": [
          "MLModel",
          "ModelArchitecture",
          "Datasource",
          "Dataset",
          "Task",
          "Method",
          "URL",
          "ReferenceLink",
          "MLModelGeneric",
          "DatasetGeneric"
        ],
        "_type": "ClassLabel"
      },
      "_type": "Sequence"
    },
    "flat_base_start": {
      "feature": {
        "dtype": "int64",
        "_type": "Value"
      },
      "_type": "Sequence"
    },
    "flat_base_end": {
      "feature": {
        "dtype": "int64",
        "_type": "Value"
      },
      "_type": "Sequence"
    },
    "flat_base_label": {
      "feature": {
        "names": [
          "MLModel",
          "ModelArchitecture",
          "Dataset",
          "Datasource",
          "Task",
          "Method",
          "URL",
          "ReferenceLink"
        ],
        "_type": "ClassLabel"
      },
      "_type": "Sequence"
    },
    "flat_plus_start": {
      "feature": {
        "dtype": "int64",
        "_type": "Value"
      },
      "_type": "Sequence"
    },
    "flat_plus_end": {
      "feature": {
        "dtype": "int64",
        "_type": "Value"
      },
      "_type": "Sequence"
    },
    "flat_plus_label": {
      "feature": {
        "names": [
          "MLModelGeneric",
          "DatasetGeneric"
        ],
        "_type": "ClassLabel"
      },
      "_type": "Sequence"
    }
  }
```
