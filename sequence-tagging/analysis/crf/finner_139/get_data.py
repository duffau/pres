import datasets
from datasets import DatasetDict
import settings as sett

datset = datasets.load_dataset(sett.DATASET_ID)


cache_directory = sett.BASE_DATASET_DIR
datset.save_to_disk(cache_directory)
print(f"Sampled dataset saved to {cache_directory}")