from flair.data import Sentence
from flair.models import SequenceTagger
from flair.nn import Classifier

from datasets import load_dataset

import time

# load tagger
tagger = Classifier.load('ner-fast')

dataset = load_dataset("eriktks/conll2003", split="train")
subset = dataset.select(range(100))  # Change the range as needed


timings = []
n_tokens = 0
n_sentences = 0
for sample in subset:
    print(n_sentences)
    n_tokens += len(sample['tokens'])
    n_sentences += 1
    text = " ".join(sample['tokens'])  # Join tokens into a full sentence
    sentence = Sentence(text)
    start_time = time.time()
    predictions = tagger.predict(sentence)
    end_time = time.time()
    elapsed_time = end_time - start_time
    timings.append(elapsed_time)

# Calculate the elapsed time
print(f"Single predict on {n_sentences} sentences and {n_tokens} tokens") 
print(f"Avg tokens per sentence {n_tokens/n_sentences}") 
print(f"Prediction time total: {sum(timings):.2f} sec")
print(f"Prediction time per sentence: {sum(timings)/n_sentences*1000:.2f} ms")
print(f"Prediction time per token: {sum(timings)/n_tokens*1000:.2f} ms")
