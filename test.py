if "__main__" == __name__:
    from ZINC_250K.dataprocess import mol_decomp_mp_ZINC_250K_pamf_pkl
    mol_decomp_mp_ZINC_250K_pamf_pkl(60)

    from pamf.pipeline import decompose_smiles

    smiles = 'Cc1[nH+]ccn1CC[S@](=O)c1cccc(F)c1'

    result = decompose_smiles(smiles)
    pass
