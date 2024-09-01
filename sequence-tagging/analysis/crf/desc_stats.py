from collections import Counter
from datasets import load_from_disk



def tag_distribution(dataset):
    tag_names = dataset.features["ner_tags"].feature.names
    train_tags = dataset['ner_tags']
    flattened_tags = [tag for sublist in train_tags for tag in sublist]
    tag_counts = Counter(flattened_tags)
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    tags, counts = zip(*sorted_tags)
    breakpoint()
    return [tag_names[i] for i in tags], counts

cache_directory = './.data/finer-139'
data = load_from_disk(cache_directory)
print(data)

tags, counts = tag_distribution(data["train"])
for tag, count in zip(tags, counts):
    print(tag, count)