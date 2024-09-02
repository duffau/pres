from datasets import load_from_disk
from utils import tag_distribution
import settings as sett



# def tag_distribution(dataset):
#     tag_names = dataset.features["ner_tags"].feature.names
#     train_tags = dataset['ner_tags']
#     flattened_tags = [tag for sublist in train_tags for tag in sublist]
#     tag_counts = Counter(flattened_tags)
#     sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
#     tags, counts = zip(*sorted_tags)
#     breakpoint()
#     return [tag_names[i] for i in tags], counts

print("Base dataset desc stats")
cache_directory = sett.FIT_DATASET_DIR
data = load_from_disk(cache_directory)
print(data)

tag_counts = tag_distribution(data["train"])
for tag, count in tag_counts:
    print(f"{count:10d}", tag)