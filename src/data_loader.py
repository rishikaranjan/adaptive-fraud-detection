import pandas as pd
from src.config import TRAIN_TRANSACTION_FILE, TRAIN_IDENTITY_FILE, TIME_COL, MERGE_IDENTITY, NROWS


def load_data():
    print("Loading transaction data...")
    transactions = pd.read_csv(TRAIN_TRANSACTION_FILE, nrows=NROWS)

    if MERGE_IDENTITY:
        print("Loading identity data...")
        identity = pd.read_csv(TRAIN_IDENTITY_FILE)
        df = transactions.merge(identity, on="TransactionID", how="left")
    else:
        df = transactions.copy()

    return df


def basic_cleaning(df):
    print("Cleaning data...")

    df = df.drop_duplicates()
    df = df.sort_values(by=TIME_COL).reset_index(drop=True)

    return df