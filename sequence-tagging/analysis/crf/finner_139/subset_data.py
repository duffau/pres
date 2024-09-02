import warnings
import settings as sett
from datasets import load_from_disk, Dataset
import settings as sett

seed = sett.SEED
subset_on_labels = sett.SUBSET_ON_LABELS
use_labels = sett.USE_LABELS
n_sample = sett.N_SAMPLE
dataset = load_from_disk(sett.BASE_DATASET_DIR)

dataset_parts = ["train", "test", "validation"]

def filter_sentences_by_tags(dataset: Dataset, tag_names: list) -> Dataset:
    tag_indices = [dataset.features["ner_tags"].feature.names.index(tag) for tag in tag_names]

    def filter_function(example):
        return any(tag in tag_indices for tag in example['ner_tags'])

    filtered_dataset = dataset.filter(filter_function)
    return filtered_dataset

if subset_on_labels:
    if use_labels is not None:
        for dataset_part in dataset_parts:
            dataset[dataset_part] = filter_sentences_by_tags(dataset[dataset_part], tag_names=use_labels)
    else:
        warnings.warn("Label subset is None")

if n_sample is not None:
    for dataset_part in dataset_parts:
        dataset[dataset_part] = dataset[dataset_part].shuffle(seed=seed).select(range(n_sample))


print("Size of datasets")
for dataset_part in dataset_parts:
    print(f"{dataset_part}: ", len(dataset[dataset_part]))


cache_directory = sett.FIT_DATASET_DIR
dataset.save_to_disk(cache_directory)
print(f"Sampled dataset saved to {cache_directory}")