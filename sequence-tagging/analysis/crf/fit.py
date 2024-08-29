from datasets import load_dataset
from itertools import chain


import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

dataset_id = "eriktks/conll2003"



def word2features(tokens, i):
    word = tokens[i]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = tokens[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(tokens)-1:
        word1 = tokens[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features


LABELS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

def sent2features(tokens):
    return [word2features(tokens, i) for i in range(len(tokens))]

def sent2labels(label_ids):
    return [LABELS[label_id] for  label_id in label_ids]


train_sents = load_dataset(dataset_id, split="train")
test_sents = load_dataset(dataset_id, split="test")


X_train = [sent2features(tokens) for tokens in train_sents["tokens"]]
y_train = [sent2labels(tag_ids) for tag_ids in train_sents["ner_tags"]]

X_test = [sent2features(tokens) for tokens in test_sents["tokens"]]
y_test = [sent2labels(tag_ids) for tag_ids in test_sents["ner_tags"]]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
    verbose=True
)


crf.fit(X_train, y_train)

labels = list(crf.classes_)
labels.remove('O')
print(labels)

y_pred = crf.predict(X_test)
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))