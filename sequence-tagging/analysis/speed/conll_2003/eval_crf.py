import settings as sett
from datasets import  load_from_disk

import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

import scipy.stats

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

import time


def word2features(tokens, pos_seq, i):
    word = tokens[i]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'pos': pos_seq[i],
        'pos[:2]': pos_seq[i][:2],
    }

    if i > 0:
        word1 = tokens[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:pos': pos_seq[i-1],
            '-1:pos[:2]': pos_seq[i-1][:2],
        })
    else:
        features['BOS'] = True


    if i < len(tokens)-1:
        word_p1 = tokens[i+1]
        features.update({
            '+1:word.lower()': word_p1.lower(),
            '+1:word.istitle()': word_p1.istitle(),
            '+1:word.isupper()': word_p1.isupper(),
            '+1:pos': pos_seq[i+1],
            '+1:pos[:2]': pos_seq[i+1][:2],
        })
    else:
        features['EOS'] = True


    return features


def tokens_to_features(tokens, pos_tags):
    return [word2features(tokens, pos_tags, i) for i in range(len(tokens))]

def ner_ids_to_names(label_ids):
    return [NER_NAMES[label_id] for label_id in label_ids]

def pos_ids_to_names(pos_ids):
    return [POS_NAMES[pos_id] for pos_id in pos_ids]


HYPERPARAMS_OPT = False

dataset = load_from_disk(sett.FIT_DATASET_DIR)
NER_NAMES = dataset["train"].features["ner_tags"].feature.names
POS_NAMES = dataset["train"].features["pos_tags"].feature.names

train_sents =  dataset["train"]
test_sents =  dataset["test"]


X_train = [tokens_to_features(tokens, pos_ids_to_names(pos_seq)) for tokens, pos_seq in zip(train_sents["tokens"], train_sents["pos_tags"])]
y_train = [ner_ids_to_names(tag_ids) for tag_ids in train_sents["ner_tags"]]

X_test = [tokens_to_features(tokens, pos_ids_to_names(pos_seq)) for tokens, pos_seq in zip(test_sents["tokens"], test_sents["pos_tags"])]
y_test = [ner_ids_to_names(tag_ids) for tag_ids in test_sents["ner_tags"]]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.002574777399011373, 
    c2=0.011416743519186584,
    max_iterations=200,
    all_possible_transitions=True,
    verbose=True
)

crf.fit(X_train, y_train)

timings = []
n_tokens = 0
n_sentences = 0
for tokens, pos_seq in zip(train_sents["tokens"], train_sents["pos_tags"]):
    n_tokens += len(tokens)
    n_sentences += 1

    start_time = time.time()
    features = tokens_to_features(tokens, pos_ids_to_names(pos_seq))
    crf.predict_single(features)
    end_time = time.time()

    elapsed_time = end_time - start_time
    timings.append(elapsed_time)

# Calculate the elapsed time
print(f"Single predict on {n_sentences} sentences and {n_tokens} tokens") 
print(f"Avg tokens per sentence {n_tokens/n_sentences}") 
print(f"Prediction time total: {sum(timings):.2f} sec")
print(f"Prediction time per sentence: {sum(timings)/n_sentences*1000:.3f} ms")
print(f"Prediction time per token: {sum(timings)/n_tokens*1000:.5f} ms")

