"""Transparent heuristic scores, with same-chemical-class WBO normalization."""

from collections import defaultdict
import math
import statistics


def fit_reference(records):
    """Fit on training-set PAMF result dictionaries, never on the test split."""
    grouped = defaultdict(list)
    for record in records:
        for row in record['candidates']:
            if row.get('wbo') is not None:
                grouped[row['chemical_class']].append(row)
    if not grouped:
        raise ValueError('No electronic features available to fit a reference')
    result = {}
    for cls, rows in grouped.items():
        result[cls] = {'count': len(rows)}
        for feature in ('wbo', 'cross_wbo'):
            values = [r[feature] for r in rows]
            if not all(math.isfinite(v) for v in values):
                raise ValueError('Reference contains nonfinite electronic features')
            result[cls][feature] = dict(mean=statistics.mean(values),
                                       std=statistics.pstdev(values))
    return result


def score_candidates(candidates, config, reference=None):
    warnings = []
    if config.mode == 'xtb' and candidates:
        if reference is None:
            reference = fit_reference([dict(candidates=candidates)])
            warnings.append('Electronic normalization uses within-molecule bond classes; scores are not calibrated across molecules.')
        for row in candidates:
            cls = row['chemical_class']
            if cls not in reference:
                raise ValueError(f'Electronic reference has no bond class {cls}')
            for feature in ('wbo', 'cross_wbo'):
                stats = reference[cls][feature]
                mean, std = float(stats['mean']), float(stats['std'])
                if not math.isfinite(mean) or not math.isfinite(std) or std < 0:
                    raise ValueError('Reference mean/std must be finite and std nonnegative')
                if std <= 1e-8:
                    row[f'z_{feature}'] = 0.0
                    warnings.append(f'{cls}/{feature}: insufficient variation; normalized electronic contribution is neutral.')
                else:
                    row[f'z_{feature}'] = (row[feature]-mean)/std
    for row in candidates:
        boundary = any(s in row['candidate_source'] for s in
                       ('BRICS_boundary', 'ring_boundary', 'functional_boundary'))
        terms = dict(rotatable=config.rotatable_weight * int(row['rotatable']),
                     boundary=config.boundary_weight * int(boundary),
                     flexibility=config.flexibility_weight * (row.get('torsion_variability') or 0.0),
                     electronic_wbo=-config.wbo_weight * row.get('z_wbo', 0.0),
                     electronic_cross=-config.cross_wbo_weight * row.get('z_cross_wbo', 0.0))
        row['score_terms'] = terms
        row['score'] = sum(terms.values())
    return sorted(set(warnings))
