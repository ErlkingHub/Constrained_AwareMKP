"""T8 single-greedy evaluation. No sampling, multistart, repair or legacy imports.

Run D_port/G_port first, then reuse --port_reference for trained best/last arms.
All historical rows must match instance bytes and reference definitions.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mkp_artifacts import sha256_file, source_provenance, write_json
from mkp_config import config_from_dict, resolve_path
from mkp_env import MKPEnv, MKPInstances
from mkp_Model import build_model
from mkp_t8_train import SCHEMA
import mkp_v6_lp as lp
from scripts.run_t8 import configuration
from scripts.t5_report import read_jsonl
from scripts.t8_report import report
from scripts.t8_data import datasets, pending_records, validate_rows


class Timings:
    """Synchronized, instrumented wall times; controller time excludes its LP."""
    def __init__(self, model, device):
        self.model, self.device = model, device
        self.lp_seconds = self.controller_seconds = self.controller_lp = 0.
        self.calls = 0

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def __enter__(self):
        self.original = lp.solve_lp_relaxation
        def solve(*args, **kwargs):
            self.sync(); tick = time.perf_counter()
            result = self.original(*args, **kwargs)
            self.sync(); self.lp_seconds += time.perf_counter() - tick; self.calls += 1
            return result
        lp.solve_lp_relaxation = solve
        def before(*args):
            self.sync(); self.actor_tick = time.perf_counter(); self.actor_lp = self.lp_seconds
        def after(*args):
            self.sync(); self.controller_seconds += time.perf_counter() - self.actor_tick
            self.controller_lp += self.lp_seconds - self.actor_lp
        actor = self.model.v6_engine.actor
        self.hooks = [actor.register_forward_pre_hook(before), actor.register_forward_hook(after)]
        self.sync(); self.began = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.sync(); self.total = time.perf_counter() - self.began
        lp.solve_lp_relaxation = self.original
        for hook in self.hooks:
            hook.remove()


def load_arm(arm, args, device):
    if arm in ('A', 'B'):
        path = Path(resolve_path(getattr(args, arm.lower() + '_checkpoint')))
        if path.name != args.tag + '.pt':
            raise ValueError('Evaluate best/last separately with matching checkpoint filenames')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        cfg = config_from_dict(checkpoint['config'])
        if checkpoint.get('schema') != SCHEMA or cfg.train.signal != ('item_pg' if arm == 'A' else 'ppo_gae'):
            raise ValueError('Wrong T8 checkpoint schema/arm')
        model = build_model(cfg); model.load_state_dict(checkpoint['model'], strict=True)
        if checkpoint['executor_config'] != asdict(model.v6_engine.exec_cfg):
            raise ValueError('Saved v6 executor constants differ from current implementation')
        provenance = {'checkpoint': str(path), 'sha256': sha256_file(path), 'update': checkpoint['update'], 'config': checkpoint['config']}
    else:
        cfg = configuration('A'); torch.manual_seed(1234); model = build_model(cfg)
        provenance = {'config': cfg.to_dict(), 'checkpoint': None, 'untrained_zero_readout': True}
    return model.to(device).eval(), provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', nargs='+', choices=('D_port', 'G_port', 'A', 'B'), default=['D_port', 'G_port'])
    parser.add_argument('--instances_dir', default='results/t5_v6')
    parser.add_argument('--port_reference', help='Completed T8 D_port/G_port output; avoids re-running these arms')
    parser.add_argument('--a_checkpoint', default='checkpoints/t8_a_jointM_s1234/best.pt')
    parser.add_argument('--b_checkpoint', default='checkpoints/t8_b_jointM_s1234/best.pt')
    parser.add_argument('--tag', choices=('best', 'last'), default='best')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--limit', type=int, choices=(0, 2), default=0, help='2 = one per M: Chu3 + synthetic2 engineering smoke')
    parser.add_argument('--output', default='results/t8_port_270')
    args = parser.parse_args(argv)
    if len(set(args.arms)) != len(args.arms):
        raise ValueError('Duplicate arm')
    args.arms = [a for a in ('D_port', 'G_port', 'A', 'B') if a in args.arms]
    source = Path(resolve_path(args.instances_dir)); history_dir = source / 'attribution'
    history_path = history_dir / 'rows.jsonl'; historical = read_jsonl(history_path)
    history_manifest = json.loads((history_dir / 'report_manifest.json').read_text(encoding='utf-8'))
    if history_manifest['status'] != 'complete' or history_manifest['smoke_only'] or sha256_file(history_path) != history_manifest['input_sha256']['rows.jsonl']:
        raise ValueError('T5 measured reference rows changed or incomplete')
    records = datasets(source, historical, bool(args.limit))
    means_path = history_dir / 'theta_means.json'; means = json.loads(means_path.read_text(encoding='utf-8'))
    g_dims = {int(m) for m in means['G']}
    if not {5, 10} <= g_dims:
        raise ValueError('T5 G requires both M5 and M10 constants')
    dest = Path(resolve_path(args.output))
    if dest.exists():
        raise FileExistsError(f'Preserve existing T8 evaluation: {dest}')
    dest.mkdir(parents=True)
    device = torch.device(args.device); torch.set_num_threads(args.threads); torch.manual_seed(1234)
    meta = {'schema': 't8_evaluation_v2_chu270', 'status': 'running', 'smoke_only': bool(args.limit), 'tag': args.tag,
        'source': source_provenance(), 'arguments': vars(args), 'arms': {}, 'historical_rows_sha256': sha256_file(history_path),
        'theta_means_sha256': sha256_file(means_path), 'bundle_sha256': sha256_file(source / 'instances_eval.npz'),
        'coverage': {'chu_dims': [5, 10, 30], 'chu_sizes': [100, 250, 500], 'formal_chu_cases': 270,
            'selected_chu_cases': sum(r[0]['dataset']=='chu' for r in records),
            'selected_synthetic_cases': sum(r[0]['dataset']=='synthetic_N100' for r in records),
            'G_available_dims': sorted(g_dims), 'missing_G_dims': sorted({5,10,30}-g_dims),
            'historical_comparisons': 'matched instance intersection only; T5 currently covers Chu M5/M10'},
        'scope': 'one greedy; historical L_v6 and F0 quoted; no sampling/search; no critic or feasibility inference',
        'timing': 'synchronized instrumented wall seconds; total includes traces; controller excludes LP nested inside actor; timings are not matched to historical hardware'}
    rows, traces = [], []
    if args.port_reference:
        prior = Path(resolve_path(args.port_reference)); pm = json.loads((prior / 'meta.json').read_text(encoding='utf-8'))
        if pm['status'] != 'complete' or pm['smoke_only'] != bool(args.limit):
            raise ValueError('Port reference status/smoke mismatch')
        for field in ('historical_rows_sha256', 'theta_means_sha256', 'bundle_sha256'):
            if pm[field] != meta[field]:
                raise ValueError('Port reference provenance mismatch')
        reused = read_jsonl(prior / 'rows.jsonl')
        if sha256_file(prior / 'rows.jsonl') != pm['rows_sha256']:
            raise ValueError('Port reference rows changed')
        rows.extend(r for r in reused if r['arm'] in ('D_port', 'G_port'))
        if {r['arm'] for r in rows} != {'D_port', 'G_port'}:
            raise ValueError('Reference must supply both baseline arms')
        validate_rows(rows, records, g_dims, complete=False)
        meta['port_reference'] = {'path': str(prior), 'meta_sha256': sha256_file(prior / 'meta.json')}
    if {'A', 'B'} & set(args.arms) and not args.port_reference and not {'D_port', 'G_port'} <= set(args.arms):
        raise ValueError('Trained comparisons require both D/G baselines or --port_reference')
    write_json(dest / 'meta.json', meta)
    # Old Chu180 baselines are valid reusable rows. Fill missing M30 D episodes
    # before A/B, while G remains restricted to the dimensions with real theta.
    planned_arms = (['D_port', 'G_port'] if args.port_reference else []) + args.arms
    planned_arms = [a for a in ('D_port','G_port','A','B') if a in planned_arms]
    print(f"Chu270 M5/M10/M30; G constants available for M={sorted(g_dims)}. Historical comparisons use matched cases only.", flush=True)
    try:
        with (dest / 'rows.jsonl').open('w', encoding='utf-8') as outcomes, (dest / 'traces.jsonl').open('w', encoding='utf-8') as trace_file:
            for r in rows:
                outcomes.write(json.dumps(r) + '\n')
            for arm in planned_arms:
                pending = pending_records(records, arm, g_dims, rows)
                meta['arms'][arm] = {'reused_rows': sum(r['arm']==arm for r in rows), 'new_rows': len(pending)}
                if not pending:
                    continue
                model, provenance = load_arm(arm, args, device); meta['arms'][arm] = provenance
                meta['arms'][arm].update(reused_rows=sum(r['arm']==arm for r in rows), new_rows=len(pending))
                for info, p, w, c in pending:
                    inst = MKPInstances(torch.tensor(p, device=device).unsqueeze(0), torch.tensor(w.T.copy(), device=device).unsqueeze(0), torch.tensor(c, device=device).unsqueeze(0))
                    override = means['G'][str(info['M'])] if arm == 'G_port' else None
                    with torch.no_grad(), Timings(model, device) as timing:
                        # D_port exercises the actual untrained new controller.
                        result = model(MKPEnv(model.ecfg), inst, decode='greedy', theta_override=override, return_trace=arm in ('A', 'B'))
                    selected = result['solution'][0].cpu().numpy()
                    remaining = c - w[selected].sum(0)
                    feasible = bool(np.all(remaining >= -1e-8))
                    maximal = not bool(((w <= remaining[None] + 1e-8).all(1) & ~selected).any())
                    if not feasible or not maximal:
                        raise ValueError('Invalid completed greedy solution')
                    objective = float(result['profit'][0]); sequence = result['sequence'][0].cpu().tolist()
                    row = {k: info[k] for k in ('dataset', 'id', 'M', 'N', 'tightness', 'reference', 'reference_kind', 'reference_proven', 'data_sha256')}
                    row.update(arm=arm, objective=objective, gap_pct=100 * (1 - objective / row['reference']), feasible=feasible,
                        maximal=maximal, selected_item_indices=sequence, time_lp_s=timing.lp_seconds,
                        time_controller_s=max(0., timing.controller_seconds - timing.controller_lp), time_total_s=timing.total,
                        lp_solve_calls=timing.calls, steps=result['steps'])
                    rows.append(row); outcomes.write(json.dumps(row) + '\n'); outcomes.flush()
                    for trace in result['trace']:
                        record = {k: row[k] for k in ('dataset', 'id', 'M', 'N', 'tightness', 'arm')}
                        record.update(step=trace['step'], tau_cap=trace['tau_cap'][0], theta={n: v[0] for n, v in trace['theta'].items()})
                        traces.append(record); trace_file.write(json.dumps(record) + '\n')
                    trace_file.flush()
                    print(f"{arm} {row['dataset']}/{row['id']}: {row['gap_pct']:.4f}% {timing.total:.2f}s", flush=True)
                del model
                if arm == 'G_port' and {'A', 'B'} & set(args.arms):
                    # Fail port reproduction before spending on learned-arm runs.
                    report(dest, rows, [], historical, bool(args.limit), records, g_dims)
        meta['rows_sha256'] = sha256_file(dest / 'rows.jsonl')
        report(dest, rows, traces, historical, bool(args.limit), records, g_dims)
        meta['status'] = 'complete'
    except Exception as exc:
        meta.update(status='failed', error=repr(exc)); raise
    finally:
        write_json(dest / 'meta.json', meta)
    return dest


if __name__ == '__main__':
    main()
