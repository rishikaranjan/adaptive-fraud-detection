def chronological_split(df, train_ratio, valid_ratio, test_ratio):
    total = len(df)

    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df = df.iloc[valid_end:].copy()

    return train_df, valid_df, test_df