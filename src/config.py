from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

TRAIN_TRANSACTION_FILE = RAW_DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY_FILE = RAW_DATA_DIR / "train_identity.csv"

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"

TRAIN_RATIO = 0.6
VALID_RATIO = 0.2
TEST_RATIO = 0.2

N_BATCHES = 10
RANDOM_STATE = 42

# Safer first run settings
MERGE_IDENTITY = False
NROWS = 100000   # first run on smaller sample