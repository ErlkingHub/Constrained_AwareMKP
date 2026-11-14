"""Canonical T8 evaluation coverage: Chu270 and the frozen synthetic128 set."""
from collections import Counter
import json
import numpy as np
from mkp_artifacts import sha256_file
from mkp_config import resolve_path
from read_mat import read_mat
from scripts.t5_export_instances import export_data_sha256

CHU_DIMS = (5, 10, 30)
CHU_SIZES = (100, 250, 500)
MATCH_FIELDS = ('data_sha256', 'reference', 'reference_kind', 'reference_proven', 'M', 'N', 'tightness')


def instance_key(row):
    return row['dataset'], str(row['id'])


def datasets(source, historical, smoke):
    """Read Chu directly; T5 is an identity witness only where it has coverage."""
    index = {instance_key(r): r for r in historical if r['arm'] == 'L'}
    records = []
    for row in read_mat(resolve_path('MKP_dataset/Benchmarks_Chu.mat')):
        if row['M'] not in CHU_DIMS or row['N'] not in CHU_SIZES:
            continue
        p = row['P'].reshape(-1); c = row['B'].reshape(-1)
        w = row['R'].reshape(row['M'], row['N']).T.copy()
        ratio = float(c.sum() / w.sum())
        info = dict(dataset='chu', id=str(row['id']), M=row['M'], N=row['N'],
            tightness=min((.25, .5, .75), key=lambda a: abs(a-ratio)), actual_capacity_ratio=ratio,
            reference=float(row['best']), reference_kind='Chu best known; not proven optimum',
            reference_proven=False, data_sha256=export_data_sha256(p, w, c))
        old = index.get(instance_key(info))
        if old is None and row['M'] in (5, 10):
            raise ValueError('Missing T5 M5/M10 identity witness')
        if old is not None and any(info[n] != old[n] for n in MATCH_FIELDS):
            raise ValueError(f"Chu T5 data/reference mismatch: {info['id']}")
        records.append((info, p, w, c))
    shapes = Counter((r[0]['M'], r[0]['N']) for r in records)
    if shapes != {(m, n): 30 for m in CHU_DIMS for n in CHU_SIZES}:
        raise ValueError(f'Chu requires all nine M/N shapes, 30 cases each: {shapes}')
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    bundle = source / 'instances_eval.npz'
    if manifest['status'] != 'complete' or sha256_file(bundle) != manifest['bundles']['eval']['sha256']:
        raise ValueError('Invalid T5 export manifest/bundle')
    with np.load(bundle, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata'].item()))
        if metadata['schema'] != 't5_instance_bundle_v1' or metadata['smoke_only']:
            raise ValueError('T8 requires the formal frozen T5 bundle even for smoke')
        for i, info in enumerate(metadata['records']):
            info = {**info, 'dataset': 'synthetic_N100'}
            old = index[instance_key(info)]
            if any(info[n] != old[n] for n in MATCH_FIELDS):
                raise ValueError('Synthetic T5 data/reference mismatch')
            records.append((info, *(archive[f'{n}_{i}'].copy() for n in ('p', 'w', 'c'))))
    for info, p, w, c in records:
        if export_data_sha256(p, w, c) != info['data_sha256']:
            raise ValueError(f"Instance hash mismatch: {instance_key(info)}")
        if p.shape != (info['N'],) or w.shape != (info['N'], info['M']) or c.shape != (info['M'],):
            raise ValueError('Instance shape mismatch')
    if Counter(r[0]['dataset'] for r in records) != {'chu': 270, 'synthetic_N100': 128}:
        raise ValueError('Incomplete Chu270/synthetic128 data')
    if len({instance_key(r[0]) for r in records}) != len(records):
        raise ValueError('Duplicate canonical instance ID')
    if smoke:
        kept, seen = [], set()
        for record in records:
            key = record[0]['dataset'], record[0]['M']
            if key not in seen:
                seen.add(key); kept.append(record)
        return kept
    return records


def arm_records(records, arm, g_dims):
    # G is an M-specific activated theta, so a missing M30 vector is not portable.
    return [r for r in records if arm != 'G_port' or r[0]['M'] in g_dims]


def validate_rows(rows, records, g_dims, complete=True):
    """Check exact IDs/metadata, including when reusing a Chu180 baseline."""
    canonical = {instance_key(r[0]): r[0] for r in records}
    index = {}
    for row in rows:
        key = row['dataset'], row['arm'], str(row['id'])
        if key in index:
            raise ValueError(f'Duplicate T8 outcome: {key}')
        if row['arm'] not in ('D_port', 'G_port', 'A', 'B'):
            raise ValueError(f"Unexpected T8 arm: {row['arm']}")
        info = canonical.get(instance_key(row))
        if info is None or any(row[n] != info[n] for n in MATCH_FIELDS):
            raise ValueError(f'Unmatched canonical instance: {key}')
        if row['arm'] == 'G_port' and row['M'] not in g_dims:
            raise ValueError('G row has no corresponding M-specific theta')
        if not row['feasible'] or not row['maximal']:
            raise ValueError(f'Invalid construction: {key}')
        if not np.isfinite(row['objective']) or not np.isclose(row['gap_pct'], 100*(1-row['objective']/row['reference']), rtol=0, atol=1e-9):
            raise ValueError(f'Invalid gap: {key}')
        index[key] = row
    if complete:
        for arm in {r['arm'] for r in rows}:
            expected = {instance_key(r[0]) for r in arm_records(records, arm, g_dims)}
            actual = {instance_key(r) for r in rows if r['arm'] == arm}
            if actual != expected:
                raise ValueError(f'Incomplete {arm}: missing {len(expected-actual)}, extra {len(actual-expected)}')
    return index


def pending_records(records, arm, g_dims, reused):
    done = {instance_key(r) for r in reused if r['arm'] == arm}
    return [r for r in arm_records(records, arm, g_dims) if instance_key(r[0]) not in done]
