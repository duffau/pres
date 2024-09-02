import settings as sett
from datasets import load_from_disk
import settings as sett
from utils import tag_distribution, format_sentence_examples


dataset = load_from_disk(sett.FIT_DATASET_DIR)

tag_counts = tag_distribution(dataset["train"], sort="alpha")
for tag, count in tag_counts:
    print(f"{count:10d}", tag)


for tag_name in sett.USE_LABELS:
    for sentence in format_sentence_examples(dataset=dataset["train"], tag_name=tag_name, seed=42):
        print(sentence)
        print("\n\n")
        user_input = input("Press Enter to see the next sentence, or type 'stop' to end: ")
        if user_input.lower() == 'stop':
            break
        