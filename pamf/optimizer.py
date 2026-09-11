"""Joint cut-set optimization, exhaustive for small candidate sets, beam otherwise."""

from .chemistry import components


def select_cuts(mol, candidates, config):
    original = components(mol, ())
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    # A pre-existing small component (e.g. a counterion) is preserved as-is.
    small_original = {frozenset(c) for c in original if len(c & heavy) < config.min_heavy_atoms}
    scores = {row['bond_idx']: row['score'] for row in candidates}

    def evaluate(cuts, final=False):
        groups = components(mol, cuts)
        sizes = [len(g & heavy) for g in groups]
        if any(n < config.min_heavy_atoms and frozenset(g) not in small_original
               for n, g in zip(sizes, groups)):
            return None
        if final and config.strict_max_size and any(n > config.max_heavy_atoms for n in sizes):
            return None
        oversize = sum(max(0, n-config.max_heavy_atoms)**2 for n in sizes)
        mean = sum(sizes)/len(sizes)
        imbalance = sum((n-mean)**2 for n in sizes)/len(sizes)/max(mean**2, 1)
        terms = dict(cut_reward=sum(scores[i] for i in cuts),
                     fragment_penalty=config.fragment_penalty*(len(groups)-len(original)),
                     oversize_penalty=config.oversize_penalty*oversize,
                     imbalance_penalty=config.imbalance_penalty*imbalance)
        value = terms['cut_reward']-sum(v for k, v in terms.items() if k != 'cut_reward')
        return value, sizes, terms

    exact = len(candidates) <= config.exact_candidate_limit
    states = [()]
    for row in sorted(candidates, key=lambda r: r['bond_idx']):
        expanded = states + [s+(row['bond_idx'],) for s in states]
        ranked = [(s, evaluate(s)) for s in expanded]
        ranked = [(s, val) for s, val in ranked if val is not None]
        ranked.sort(key=lambda entry: (-entry[1][0], len(entry[0]), entry[0]))
        states = [s for s, _ in (ranked if exact else ranked[:config.beam_width])]
        if not exact and () not in states:
            states.append(())  # Always retain the uncut baseline.
    final = [(s, evaluate(s, final=True)) for s in states]
    final = [(s, v) for s, v in final if v is not None]
    if not final:
        qualifier = 'No feasible partition exists' if exact else 'Beam search found no feasible partition'
        raise ValueError(qualifier + '; protected units/size constraints may be incompatible')
    final.sort(key=lambda entry: (-entry[1][0], len(entry[0]), entry[0]))
    cuts, (value, sizes, terms) = final[0]
    return cuts, dict(method='exhaustive' if exact else 'beam', optimality_proven=exact,
                      objective=value, objective_terms=terms, fragment_heavy_atom_counts=sorted(sizes),
                      size_limit_satisfied=all(n <= config.max_heavy_atoms for n in sizes),
                      preexisting_small_components=len(small_original),
                      candidate_count=len(candidates), beam_width=None if exact else config.beam_width)
