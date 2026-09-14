"""Count fragment graph nodes using the same canonicalization as fragment_vq."""
import json
from collections import Counter
from pathlib import Path

from autoencoder.fragment_vq import canonical_fragment


def describe(hist):
    total = sum(hist.values())
    def percentile(p):
        cumulative = 0
        for n, count in sorted(hist.items()):
            cumulative += count
            if cumulative >= p * total:
                return n
    return dict(count=total, min=min(hist), max=max(hist),
                mean=sum(n*c for n, c in hist.items())/total,
                median=percentile(.5), p90=percentile(.9),
                p95=percentile(.95), p99=percentile(.99),
                over_70=sum(c for n, c in hist.items() if n > 70),
                histogram=dict(sorted(hist.items())),
                bins={f'{lo}-{hi}':sum(c for n,c in hist.items() if lo <= n <= hi)
                      for lo,hi in [(1,5),(6,10),(11,15),(16,20),(21,25),(26,30),(31,40),(41,50),(51,70)]})


def summarize(records, cache):
    unique, weighted, heavy, canonical = Counter(), Counter(), Counter(), Counter()
    dummy_fragments = explicit_h_fragments = 0
    for smiles, frequency in records.items():
        if smiles not in cache:
            normalized, mol = canonical_fragment(smiles)
            cache[smiles] = (normalized, mol.GetNumAtoms(),
                            sum(a.GetAtomicNum() > 1 for a in mol.GetAtoms()),
                            sum(a.GetAtomicNum() == 0 for a in mol.GetAtoms()),
                            sum(a.GetAtomicNum() == 1 for a in mol.GetAtoms()))
        normalized, nodes, heavy_atoms, dummy, explicit_h = cache[smiles]
        unique[nodes] += 1
        weighted[nodes] += frequency
        heavy[heavy_atoms] += 1
        canonical[normalized] += frequency
        dummy_fragments += dummy > 0
        explicit_h_fragments += explicit_h > 0
    canonical_hist = Counter()
    seen = set()
    for s in records:
        normalized, nodes, *_ = cache[s]
        if normalized not in seen:
            canonical_hist[nodes] += 1
            seen.add(normalized)
    return dict(raw_unique=describe(unique), occurrence_weighted=describe(weighted),
                canonical_unique=describe(canonical_hist), heavy_atoms_raw_unique=describe(heavy),
                fragments_with_dummy=dummy_fragments,
                fragments_with_explicit_h_nodes=explicit_h_fragments)


def main():
    cache, datasets = {}, {}
    for split in ('train', 'test'):
        path = Path(f'stru_data_ZINC_250K_pamf_{split}.json')
        datasets[split] = json.loads(path.read_text(encoding='utf-8'))['pamf']
    merged = Counter(datasets['train']) + Counter(datasets['test'])
    result = {split:summarize(data, cache) for split,data in
              [*datasets.items(), ('combined', merged)]}
    result['raw_overlap'] = len(datasets['train'].keys() & datasets['test'].keys())
    result['definition'] = 'RDKit nodes after fragment_vq canonicalization; includes dummy atoms, does not expand implicit H. Weights are JSON frequencies.'
    output = Path('autoencoder/fragment_atom_statistics.json')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    for split in ('train','test','combined'):
        print(split, json.dumps({k:({x:y for x,y in v.items() if x != 'histogram'} if isinstance(v,dict) else v) for k,v in result[split].items()}))
    print('raw_overlap', result['raw_overlap'])


if __name__ == '__main__':
    main()
