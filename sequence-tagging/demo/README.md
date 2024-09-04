# Demonstration of Conditional Random Field (CRF)

This project demonstrates how to train a Conditional Random Field (CRF) model on the CoNLL-2003 dataset, a widely used dataset for Named Entity Recognition (NER). The model is implemented using the [scikit-learn-crfsuite](https://sklearn-crfsuite.readthedocs.io/en/latest/) package, which wraps the [CRFsuite library](http://www.chokkan.org/software/crfsuite/) for sequence modeling tasks.

## Setup

To set up the environment and install dependencies:

```bash
$ python -m venv .venv
$ .venv/bin/activate
$ .venv/bin/pip install -r requirements.txt
```

Ensure that you have the required Python packages installed, such as `datasets`, `sklearn-crfsuite`, and others listed in the `requirements.txt` file.

## Dataset

The model is trained on the [CoNLL-2003](https://huggingface.co/datasets/eriktks/conll2003) dataset, which consists of English text labeled for four types of named entities:
- **PER**: Person names (e.g., "John", "Elon Musk")
- **ORG**: Organization names (e.g., "Google", "United Nations")
- **LOC**: Locations, including countries, cities, etc. (e.g., "Paris", "Germany")
- **MISC**: Miscellaneous entities, such as nationalities, events, etc. (e.g., "German", "Olympics")

Each token (word) in the dataset is annotated with one of these entity types or marked as `O` for tokens that are not part of any named entity.

### Example Sentence from the Dataset:

A sample sentence with tokens and corresponding NER tags from the CoNLL-2003 dataset might look like this:

**Sentence:**
```
["Germany", "won", "the", "FIFA", "World", "Cup", "in", "2014", "."]
```

**Tags:**
```
["B-LOC", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O", "O", "O"]
```

Here, "Germany" is tagged as `B-LOC` (beginning of a location entity), and "FIFA World Cup" is tagged as a miscellaneous entity (`B-MISC`, `I-MISC`).

#### Tagging Scheme:
The dataset uses the **IOB** (Inside-Outside-Beginning) tagging scheme:
- `B-` denotes the beginning of a named entity.
- `I-` indicates the continuation of the same named entity.
- `O` is used for words that do not belong to any named entity.

## Run CRF Fitting and Validation

To run the CRF model training and validation, execute the following command:

```bash
$ (.venv) python fit.py
```

The script loads the CoNLL-2003 dataset, extracts features from the tokens (such as word shape, previous/next words), and trains a CRF model. After training, it evaluates the model on the test set and prints a classification report. 

## CRFsuite Links

For further information on CRFsuite and its usage:

- [scikit-learn-crfsuite Documentation](https://sklearn-crfsuite.readthedocs.io/en/latest/): A Python library for training and using CRF models, based on the CRFsuite C++ library.
- [CRFsuite Documentation](http://www.chokkan.org/software/crfsuite/): The official documentation of the underlying CRF library used by scikit-learn-crfsuite.

This demo script demonstrates how CRFs can be effectively used for sequence labeling tasks such as Named Entity Recognition (NER).