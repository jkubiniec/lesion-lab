import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('large_analysis',ROOT/'analyze_large_run.py')
analysis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def sample(identity=1,slot=0,correct=3,injured=False):
    return {'id':identity,'slot':slot,'injured':injured,'disabled_count':int(injured),'episodes':4,'correct':correct,
            'go_trials':2,'go_correct':min(correct,2),'nogo_trials':2,'nogo_correct':max(0,correct-2),
            'target_per_trial':'1200','go_per_trial':'1100'}


def assay(samples):
    totals={key:sum(s[key] for s in samples) for key in ('episodes','correct','go_trials','go_correct','nogo_trials','nogo_correct')}
    return {'samples':len(samples),'trials':4,'per_sample':samples,**totals}


class AnalysisTests(unittest.TestCase):
    def test_paired_immediate_excludes_uninjured_and_replaced_identities(self):
        before=assay([sample(1,0),sample(2,1),sample(3,2)])
        after=assay([sample(1,0,1,True),sample(2,1,4,False),sample(99,2,4,True)])
        row=analysis.paired_effects(before,after,repeats=100)[0]
        self.assertEqual(row['paired_individuals'],1)
        self.assertEqual(row['mean_delta'],-.5)
        self.assertEqual(row['individuals_worsened'],1)
        self.assertEqual(row['bootstrap_ci95_low'],-.5)
        self.assertEqual(row['inference_scope'],'within_one_evolved_world_only')

    def test_changed_actions_and_trial_gains_are_descriptive_not_new_individuals(self):
        before=sample(correct=2)
        before.update(correct_per_trial='1010',action_per_trial='1001')
        after=sample(correct=2,injured=True)
        after.update(correct_per_trial='0110',action_per_trial='2201')
        rows=analysis.paired_effects(assay([before]),assay([after]),repeats=100)
        self.assertEqual(rows[0]['paired_individuals'],1)
        self.assertEqual(rows[0]['paired_trials'],4)
        self.assertEqual(rows[0]['trials_gained'],1)
        self.assertEqual(rows[0]['trials_lost'],1)
        self.assertEqual(rows[0]['actions_changed'],2)
        self.assertEqual(rows[0]['action_change_fraction'],.5)
        self.assertEqual(rows[1]['paired_trials'],2)
        self.assertEqual(rows[2]['paired_trials'],2)

    def test_unmatched_task_streams_and_slots_are_rejected(self):
        before=assay([sample()])
        after=assay([sample(injured=True)])
        after['per_sample'][0]['go_per_trial']='1010'
        with self.assertRaisesRegex(ValueError,'unmatched task'):
            analysis.paired_effects(before,after,repeats=100)
        after=assay([sample(slot=1,injured=True)])
        with self.assertRaisesRegex(ValueError,'different task slots'):
            analysis.paired_effects(before,after,repeats=100)

    def test_interval_rates_use_counter_differences_not_lifetime_rates(self):
        before={'decisions':100,'episodes':10,'correct':8,'births':2}
        after={'decisions':200,'episodes':20,'correct':10,'births':5}
        row=analysis.interval_progress([before,after],'warmup')[0]
        self.assertEqual(row['accuracy'],.2)
        self.assertEqual(row['births_interval'],3)
        self.assertIsNone(row['go_accuracy'])

    def test_resume_prefix_stitches_overlap_and_detects_counter_conflicts(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            prefix=[{'decisions':0,'episodes':0,'correct':0,'births':0}, {'decisions':20,'episodes':10,'correct':4,'births':3}]
            resumed=[prefix[-1],{'decisions':40,'episodes':20,'correct':12,'births':6}]
            for name,rows in [('prefix_warmup.jsonl',prefix),('warmup.jsonl',resumed)]:
                (run/name).write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            rows=analysis.warmup_progress(run)
            self.assertEqual([r['decisions'] for r in rows],[0,20,40])
            intervals=analysis.interval_progress(rows,'warmup')
            self.assertEqual([r['accuracy'] for r in intervals],[.4,.8])
            resumed[0]={**resumed[0],'births':7}
            (run/'warmup.jsonl').write_text('\n'.join(json.dumps(r) for r in resumed)+'\n')
            with self.assertRaisesRegex(ValueError,'Conflicting prefix'):
                analysis.warmup_progress(run)

    def test_bootstrap_is_reproducible_and_zero_sample_is_explicit(self):
        self.assertEqual(analysis.bootstrap_mean([-.5,.2,.1],100),analysis.bootstrap_mean([-.5,.2,.1],100))
        self.assertEqual(analysis.bootstrap_mean([]),(None,None,None))

    def test_complete_analysis_writes_valid_svgs_and_portable_tables(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)/'run';run.mkdir()
            out=Path(folder)/'results';figs=Path(folder)/'figures'
            base=assay([sample(1,0),sample(2,1)])
            injured=assay([sample(1,0,2,True),sample(2,1)])
            before={'injured_initial':1,'injured_ids':[1],'injured_alive':1,'births':5,'decisions':100}
            final={'injured_initial':1,'injured_alive':0,'births':6,'injured_birth_replacements':1,
                   'individuals':[{'id':3,'alive':1},{'id':2,'alive':1}]}
            manifest={'status':'complete','population':2,'hidden':8,'seed':42,'warmup_final':{'decisions':100,'births':5},
                      'arms':{'designated_region':{'immediate':before,'final':final,'immediate_assay':injured,'final_assay':base}}}
            (run/'manifest.json').write_text(json.dumps(manifest))
            (run/'baseline_assay.json').write_text(json.dumps(base))
            (run/'warmup.jsonl').write_text(json.dumps({'decisions':0,'episodes':0,'correct':0,'births':0})+'\n'+json.dumps({'decisions':100,'episodes':8,'correct':4,'births':5})+'\n')
            control={**base,'cue_mode':'erased'}
            (run/'baseline_cue_erased_assay.json').write_text(json.dumps(control))
            result=analysis.analyze(run,out,figs,repeats=100)
            self.assertEqual(result['cohort_retention'][0]['original_injured_alive'],0)
            self.assertFalse(result['cohort_retention'][0]['repair_implemented'])
            self.assertNotIn(folder,(out/'summary.json').read_text())
            self.assertEqual(len(list(figs.glob('*.svg'))),8)
            for p in figs.glob('*.svg'):
                self.assertEqual(ET.parse(p).getroot().tag,'{http://www.w3.org/2000/svg}svg')
            self.assertTrue((out/'paired_effects.csv').exists())
            self.assertEqual(result['cue_control_effects'][0]['paired_individuals'],2)

    def test_direct_placement_comparison_pairs_only_shared_injured_identities(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);out=run/'results';figs=run/'figs'
            base=assay([sample(1,0,4),sample(2,1,4)])
            designated=assay([sample(1,0,3,True),sample(2,1,4,False)])
            scattered=assay([sample(1,0,2,True),sample(2,1,0,True)])
            arms={name:{'immediate':{'injured_initial':1},'final':{'injured_alive':1},'immediate_assay':a,'final_assay':a}
                  for name,a in [('designated_region',designated),('random',scattered)]}
            manifest={'status':'complete','population':2,'hidden':8,'arms':arms}
            (run/'manifest.json').write_text(json.dumps(manifest))
            (run/'baseline_assay.json').write_text(json.dumps(base))
            result=analysis.analyze(run,out,figs,repeats=100)
            row=result['paired_placement_effects'][0]
            self.assertEqual(row['comparison'],'scattered_minus_designated')
            self.assertEqual(row['paired_individuals'],1)
            self.assertEqual(row['mean_delta'],-.25)
            self.assertTrue((out/'placement_comparison.csv').exists())

    def test_relevance_diagram_separates_emergence_units_and_injury_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'relevance.svg'
            analysis.relevance_figure(path)
            root=ET.parse(path).getroot()
            words=' '.join(root.itertext())
            for required in ('can self-copying emerge', 'Its counts are not equivalent to BFF interactions',
                             'Frozen weights; no births in the assay', 'Distinguishes replacement from repair',
                             'birth is engineered', 'frontal specialization'):
                self.assertIn(required,words)

    def test_incomplete_run_refuses_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);(run/'manifest.json').write_text('{"status":"running"}')
            with self.assertRaisesRegex(ValueError,'incomplete'):
                analysis.analyze(run,run/'out',run/'figs')


if __name__=='__main__':unittest.main()
