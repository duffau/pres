import time
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from datasets import load_dataset

# Load the tokenizer and model
model_name = "kamalkraj/bert-base-cased-ner-conll2003"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Load the dataset (train split)
dataset = load_dataset("eriktks/conll2003", split="train")

# Create a pipeline for named entity recognition (NER)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Select a subset of the dataset for faster processing (optional)
subset = dataset  # Change the range as needed

# Timing the prediction

# Perform NER on the subset of sentences
timings = []
n_tokens = 0
n_sentences = 0
for sample in subset:
    n_tokens += len(sample['tokens'])
    n_sentences += 1
    text = " ".join(sample['tokens'])  # Join tokens into a full sentence
    start_time = time.time()
    predictions = ner_pipeline(text)
    end_time = time.time()
    elapsed_time = end_time - start_time
    timings.append(elapsed_time)

# Calculate the elapsed time
print(f"Single predict on {n_sentences} sentences and {n_tokens} tokens") 
print(f"Prediction time total: {sum(timings):.2f} sec")
print(f"Prediction time per sentence: {sum(timings)/n_sentences*1000:.2f} ms")
print(f"Prediction time per token: {sum(timings)/n_tokens*1000:.2f} ms")
