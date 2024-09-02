import settings as sett
from datasets import  load_from_disk
import re

import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import scipy.stats

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


RE_NUM = re.compile(r"(\d+[\d,.]*)|([,.]\d+)")
RE_DIGITS = re.compile(r"\d")

SIGNAL_WORDS = set([
    # SharePrice
    "share",
    "stock",
    "price",
    # Revenue
    "sales",
    # GoodWill
    "goodwill"
])

def word2features(tokens, i):
    word = tokens[i]
    left_context = [word.lower() for word in tokens[(max(i-3, 0)):i]]
    right_context = [word.lower() for word in tokens[(i+1):(i+4)]]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        "[NUM]": True if RE_NUM.fullmatch(word) else False,
        "[SHAPE]": RE_DIGITS.sub('X', word),
        'word_left_window': word 
    }

    for lword in left_context:
        if lword in SIGNAL_WORDS:
            features["left_context"] = lword
    if i > 0:
        word1 = tokens[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.is_dollar': word1 == "$",
        })
    else:
        features['BOS'] = True

    if i > 1:
        word_m2 = tokens[i-2].lower()
        features.update({
            '-2:share': word_m2 == "share",
            '-2:stock': word_m2 == "stock",
            '-2:price': word_m2 == "price",
            '-2:price': word_m2 == "price",
            '-2:sales': word_m2 == "sales",
            '-2:goodwill': word_m2 == "goodwill"
        })

    # if i > 3:
    #     word_m3 = tokens[i-3].lower()
    #     word_m4 = tokens[i-4].lower()
    #     features.update({
    #         '-4:share': word_m4 == "share",
    #         '-4:stock': word_m4 == "stock",
    #         '-3:price': word_m3 == "price",
    #         '-4:price': word_m4 == "price",
    #         '-3:sales': word_m3 == "sales",
    #     })


    if i < len(tokens)-1:
        word_p1 = tokens[i+1]
        features.update({
            '+1:word.lower()': word_p1.lower(),
            '+1:word.istitle()': word_p1.istitle(),
            '+1:word.isupper()': word_p1.isupper(),
        })
    else:
        features['EOS'] = True



    # if i < len(tokens)-2:
    #     word_p2 = tokens[i+2].lower()
    #     features.update({
    #         '+2:share': word_p2 == "share",
    #         '+2:goodwill': word_p2 == "goodwill"
    #     })

    # if i < len(tokens)-3:
    #     word_p3 = tokens[i+3].lower()
    #     features.update({
    #         '+1:per': word_p3 == "per",
    #         '+2:share': word_p3 == "share",
    #         '+2:goodwill': word_p3 == "goodwill"
    #     })

    return features



def sent2features(tokens):
    return [word2features(tokens, i) for i in range(len(tokens))]

def sent2labels(label_ids, use_labels = None):
    if use_labels is None:
        return [LABELS[label_id] for label_id in label_ids]
    else:
        return [LABELS[label_id] if label_id in use_labels else "O" for label_id in label_ids]


HYPERPARAMS_OPT = False

dataset = load_from_disk(sett.FIT_DATASET_DIR)
LABELS = dataset["train"].features["ner_tags"].feature.names
if sett.USE_LABELS is not None:
    USE_LABELS = set([LABELS.index(label_name) for label_name in sett.USE_LABELS])
else:
    USE_LABELS = None

train_sents =  dataset["train"]
test_sents =  dataset["test"]


X_train = [sent2features(tokens) for tokens in train_sents["tokens"]]
y_train = [sent2labels(tag_ids, USE_LABELS) for tag_ids in train_sents["ner_tags"]]

X_test = [sent2features(tokens) for tokens in test_sents["tokens"]]
y_test = [sent2labels(tag_ids, USE_LABELS) for tag_ids in test_sents["ner_tags"]]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.4487, 
    c2=0.00237,
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

if HYPERPARAMS_OPT:
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
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
