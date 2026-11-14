"""Chu270 coverage, baseline migration and matched-subset reporting contracts."""
from collections import Counter
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from mkp_config import Config
from mkp_env import MKPEnv, MKPInstances
from mkp_Model import build_model
from scripts.t5_report import read_jsonl
from scripts.t8_data import datasets, pending_records, validate_rows, instance_key
from scripts.t8_report import report

ROOT = Path(__file__).resolve().parents[1]


class CoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source = ROOT/'results/t5_v6'
        if not (cls.source/'instances_eval.npz').exists():
            raise unittest.SkipTest('Frozen T5 data required')
        cls.history = read_jsonl(cls.source/'attribution/rows.jsonl')
        cls.records = datasets(cls.source, cls.history, False)
        cls.g_dims = {5,10}

    def test_all_nine_shapes_and_training_protocol(self):
        chu = [r[0] for r in self.records if r[0]['dataset']=='chu']
        self.assertEqual(len(chu),270)
        self.assertEqual(len({instance_key(r) for r in chu}),270)
        self.assertEqual(Counter((r['M'],r['N']) for r in chu),{(m,n):30 for m in (5,10,30) for n in (100,250,500)})
        self.assertEqual(sum(r[0]['dataset']=='synthetic_N100' for r in self.records),128)
        cfg=Config()
        self.assertEqual(cfg.eval.dims,[5,10,30])
        self.assertEqual(cfg.env.train_dims,[5,10]);self.assertEqual(cfg.train.val_dims,[5,10])

    def baselines(self):
        return [{**r,'time_controller_s':0.,'arm':{'D':'D_port','G':'G_port'}[r['arm']]} for r in self.history if r['arm'] in ('D','G')]

    def test_chu180_reference_fills_exactly_the_missing_m30(self):
        rows=self.baselines()
        validate_rows(rows,self.records,self.g_dims,complete=False)
        missing=pending_records(self.records,'D_port',self.g_dims,rows)
        self.assertEqual(len(missing),90)
        self.assertTrue(all(r[0]['M']==30 for r in missing))
        self.assertEqual(pending_records(self.records,'G_port',self.g_dims,rows),[])
        with self.assertRaisesRegex(ValueError,'Incomplete D_port'):
            validate_rows(rows,self.records,self.g_dims)

    @staticmethod
    def outcome(info,arm,gap):
        return {**info,'arm':arm,'objective':info['reference']*(1-gap/100),'gap_pct':gap,
            'feasible':True,'maximal':True,'selected_item_indices':[], 'steps':1,
            'time_lp_s':0.,'time_controller_s':0.,'time_total_s':0.}

    def test_report_uses_matched_denominators(self):
        rows=self.baselines();traces=[]
        for info,*_ in self.records:
            if info['M']==30:
                rows.append(self.outcome(info,'D_port',3.))
            for arm in ('A','B'):
                gap=(20. if arm=='A' else 30.) if info['M']==30 else (.4 if arm=='A' else .5)
                rows.append(self.outcome(info,arm,gap))
                if info['dataset']=='chu':
                    m=info['M']
                    theta={f:1. for f in ('profit_exponent','scarcity_exponent','bottleneck_weight','temperature',
                        'opportunity_weight','reserve_weight','lp_primal_weight','lp_dual_weight')}
                    theta.update(resource_prices=[1.]*m,resource_interactions=np.zeros((m,m)).tolist(),companion_weights=[0.,0.,0.])
                    traces.append({**info,'arm':arm,'step':0,'tau_cap':.5,'theta':theta})
        with tempfile.TemporaryDirectory() as directory:
            dest=Path(directory)
            report(dest,rows,traces,self.history,False,self.records,self.g_dims)
            def csv_rows(name):
                with (dest/name).open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))
            totals={r['arm']:int(r['count']) for r in csv_rows('summary.csv') if r['dataset']=='chu' and r['M']=='all'}
            self.assertEqual(totals,{'D_port':270,'G_port':180,'A':270,'B':270,'L_v6':180})
            pairs={r['control']:int(r['paired_count']) for r in csv_rows('paired_coverage.csv') if r['dataset']=='chu' and r['arm']=='A'}
            self.assertEqual(pairs,{'D_port':270,'G_port':180,'L_v6':180,'B':270})
            recovery=next(r for r in csv_rows('recovery.csv') if r['dataset']=='chu' and r['arm']=='A' and r['M']=='all')
            self.assertEqual(int(recovery['count']),180);self.assertEqual(int(recovery['arm_count']),270)
            self.assertAlmostEqual(float(recovery['arm_mean_gap_pct']),.4)
            checks=json.loads((dest/'reproduction.json').read_text())['checks']
            d=next(r for r in checks if r['arm']=='D_port' and r['dataset']=='chu')
            self.assertEqual((d['count'],d['evaluated_count'],d['not_covered_by_T5']),(180,270,90))
            self.assertTrue(d['formal_verified'])
            missing=next(r for r in csv_rows('recovery.csv') if r['dataset']=='chu' and r['M']=='30')
            self.assertEqual(missing['count'],'0');self.assertEqual(missing['recovery_fraction'],'')

    def test_rejects_duplicates_and_changed_instance_metadata(self):
        rows=self.baselines()
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            validate_rows(rows+[rows[0]],self.records,self.g_dims,complete=False)
        changed=copy.deepcopy(rows);changed[0]['data_sha256']='different'
        with self.assertRaisesRegex(ValueError,'Unmatched'):
            validate_rows(changed,self.records,self.g_dims,complete=False)

    def test_actual_m30_single_greedy(self):
        info,p,w,c=next(r for r in self.records if r[0]['dataset']=='chu' and r[0]['M']==30)
        inst=MKPInstances(torch.tensor(p).unsqueeze(0),torch.tensor(w.T.copy()).unsqueeze(0),torch.tensor(c).unsqueeze(0))
        cfg=Config();cfg.model.executor=cfg.model.theta_family='v6';cfg.model.ctrl_residual_lp_features=True
        cfg.model.embed_dim=16;cfg.model.num_heads=2;cfg.model.encoder_layers=cfg.model.controller_layers=1;cfg.model.edge_hidden=8
        model=build_model(cfg).eval()
        with torch.no_grad():out=model(MKPEnv(cfg.env),inst,decode='greedy')
        selected=out['solution'][0].numpy();remaining=c-w[selected].sum(0)
        self.assertTrue(np.isfinite(float(out['profit'][0])))
        self.assertTrue((remaining>=0).all())
        self.assertFalse(((w<=remaining).all(1)&~selected).any())


if __name__=='__main__':unittest.main()
