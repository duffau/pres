import re
import datasets
import sklearn_crfsuite
from sklearn_crfsuite import metrics


def tokens_to_features(tokens, i):
    features = {
        "bias": 1.0,
        "word": tokens[i].lower(),
        "prev_word": tokens[i - 1] if i>0 else "BOS",
        "next_word": tokens[i + 1] if i>len(tokens) else "EOS",
        "shape": re.sub(r"\d", "X", tokens[i]),
    }
    return features


def load_X_y(dataset_id="eriktks/conll2003", split="train"):
    data = datasets.load_dataset(dataset_id)
    sentences = data[split]["tokens"]
    labels = data[split]["ner_tags"]
    label_names = data[split].features["ner_tags"].feature.names
    
    X, y = [], []
    for sentence, label_seq in zip(sentences, labels):
        X.append([tokens_to_features(sentence, i) for i in range(len(sentence))])
        y.append([label_names[label_id] for label_id in label_seq])
    return X, y


X, y = load_X_y(split="train")
crf = sklearn_crfsuite.CRF(
    algorithm="lbfgs",
    c1=0.5,
    c2=0.01,
    all_possible_states=True,
    all_possible_transitions=True,
    max_iterations=100,
    verbose=True
)
crf.fit(X, y)


label_names = set([tag for tags in y for tag in tags])
label_names.remove("O")
label_names = list(label_names)


X_test, y_test = load_X_y(split="test")
y_pred = crf.predict(X_test)
print(metrics.flat_classification_report(y_test, y_pred, labels=label_names))

print("train n sentence:", len(X))
print("train n tokens:", len([label for label_seq in y for label in label_seq]))
print("test n sentence:", len(y_test))
print("test n tokens:", len([label for label_seq in y_test for label in label_seq]))
