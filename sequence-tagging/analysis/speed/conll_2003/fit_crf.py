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

if HYPERPARAMS_OPT:
    params_space = {
        'c1': scipy.stats.expon(scale=0.1),
        'c2': scipy.stats.expon(scale=0.1),
    }

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted', labels=labels)

    # search
    rs = RandomizedSearchCV(crf, params_space,
                            cv=3,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=50,
                            scoring=f1_scorer,
                            random_state=sett.SEED)
    rs.fit(X_train, y_train)

    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)
    print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

    crf = rs.best_estimator_
    y_pred = crf.predict(X_test)
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))
