import datasets
from datasets import DatasetDict

N_TRAIN_SAMPLE = 1000
N_VAL_SAMPLE = 1000
N_TEST_SAMPLE = 1000
SEED = 42

finer = datasets.load_dataset("nlpaueb/finer-139")

sampled_train = finer['train'].shuffle(seed=SEED).select(range(N_TRAIN_SAMPLE))
sampled_validation = finer['validation'].shuffle(seed=SEED).select(range(N_VAL_SAMPLE))
sampled_test = finer['test'].shuffle(seed=SEED).select(range(N_TEST_SAMPLE))


sampled_dataset = DatasetDict({
    'train': sampled_train,
    'validation': sampled_validation,
    'test': sampled_test
})


cache_directory = './.data/finer-139'
sampled_dataset.save_to_disk(cache_directory)
print(f"Sampled dataset saved to {cache_directory}")