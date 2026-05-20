import pyarrow.parquet as pq

def data_from_ZINC_250K():
    train_df = _parquet2smiles('data/ZINC_250K_train.parquet')
    test_df = _parquet2smiles('data/ZINC_250K_test.parquet')

    train = train_df['smiles'].tolist()
    test = test_df['smiles'].tolist()
    all = []
    for target in train:
        all.append(target)
    for target in test:
        all.append(target)

    return train, test, all

def _parquet2smiles(path):
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    df = table.to_pandas()
    return df