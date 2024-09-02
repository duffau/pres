DATASET_ID = "nlpaueb/finer-139"
N_SAMPLE = None
SEED = 42

BASE_DATASET_DIR = f'./.data/base/{DATASET_ID}'
FIT_DATASET_DIR = f'./.data/fit/{DATASET_ID}'

SUBSET_ON_LABELS = True

USE_LABELS = [
    "B-SharePrice",
    "B-Revenues",
    "B-Goodwill",
    "B-GoodwillImpairmentLoss",
    "B-LongTermDebt",
    "B-DebtInstrumentFaceAmount",
    "B-DebtInstrumentMaturityDate",
    "I-DebtInstrumentMaturityDate",
    "B-LineOfCreditFacilityMaximumBorrowingCapacity",
    "B-LineOfCreditFacilityRemainingBorrowingCapacity"
]
