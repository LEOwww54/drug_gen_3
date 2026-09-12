if "__main__" == __name__:
    from ZINC_250K.dataloader import data_from_ZINC_250K as loader
    from pamf.pipeline import fragment_smiles as decompose
    smiles = loader()[1]
    smiles = smiles [:100]
    result = []
    for i in smiles:
        result.append(decompose(i))
    print()