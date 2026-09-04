#!/usr/bin/env python3
"""Corrected Stage C7.1 frozen-baseline memory scale audit."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.extract_vggt_features import load_config
from scripts.audit_stage_c5_v4_branches import load_expected_cache
from viewtoken.oracle import (PinholeIntrinsics, build_memory_id, build_visibility_masks, calibrate_candidate_depth_scale, build_disturbance_states_from_sampled_h0, load_gt_poses_for_view_ids, load_point_cloud, recover_branch_points_by_view, sample_views, scale_protocols_for_branch, spearman_rank_correlation, state_metrics, uncertainty_intervals_overlap)
from viewtoken.oracle.metrics import nearest_neighbor_squared_distances, sample_points

BRANCHES=("B","C","D")

def stats(values):
 a=np.asarray(values,dtype=float); return {"mean":float(a.mean()),"std":float(a.std()),"min":float(a.min()),"max":float(a.max()),"positive_fraction":float((a>0).mean())}
def outliers(points,target):
 d=nearest_neighbor_squared_distances(points,target).sqrt(); return {"0.1":float((d>.1).float().mean()),"0.2":float((d>.2).float().mean())}
def rank(values,candidates): return [c for _,c in sorted(zip(values,candidates),reverse=True)]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,default=Path('configs/oracle_stage_c5_v4_smoke.yaml')); ap.add_argument('--quota',type=int,default=3000); ap.add_argument('--reconstruction-sample-seed',type=int,default=0); args=ap.parse_args(); cfg=load_config(args.config)
 obs_paths=[Path(x) for x in cfg['observed-views']]; obs=[p.stem for p in obs_paths]; cand_paths=[Path(x) for x in cfg['candidate-views']]; cands=[p.stem for p in cand_paths]; cache_root=Path(cfg['output-dir'])/'reconstructions'; mem=build_memory_id(cfg['scene-id'],obs); pose_dir=Path(cfg['posed-image-dir'])
 base=load_expected_cache(cache_root/mem,obs_paths,cfg); base_poses=load_gt_poses_for_view_ids(obs,pose_dir); native={b:recover_branch_points_by_view(base,base_poses,obs,b) for b in BRANCHES}; base_scale=native['C'][1]['per_cache_depth_scale']
 shared_h0={b:torch.cat(sample_views(recover_branch_points_by_view(base,base_poses,obs,b,depth_scale=base_scale)[0],args.quota,args.reconstruction_sample_seed)) for b in BRANCHES}
 source=load_point_cloud(Path(cfg['target-points']),point_stride=int(cfg.get('point-stride',6))); source_count=int(source.shape[0]); matrix=torch.tensor(np.loadtxt(cfg['intrinsics'],dtype=np.float32))[:3,:3]
 with Image.open(obs_paths[0]) as im: intr=PinholeIntrinsics.from_matrix(matrix,im.width,im.height)
 calibrations={}; caches={}; poses={}
 for cid,path in zip(cands,cand_paths):
  caches[cid]=load_expected_cache(cache_root/f'{mem}__plus__{cid}',obs_paths+[path],cfg); poses[cid]=load_gt_poses_for_view_ids(obs+[cid],pose_dir); calibrations[cid]=calibrate_candidate_depth_scale(base.depth,caches[cid].depth,3,base_scale)
  if calibrations[cid]['status']!='calibrated': raise RuntimeError(f"blocked_inconsistent_observed_depth_scale: {cid}")
 runs=[]; h0_fingerprints={b:{"shape":list(shared_h0[b].shape),"sum":float(shared_h0[b].double().sum())} for b in BRANCHES}
 for target_size,target_seeds in ((12000,range(5)),(50000,(0,))):
  for target_seed in target_seeds:
   target=sample_points(source,target_size,seed=target_seed,method='hash')
   for cid in cands:
    masks=build_visibility_masks(target,[poses[cid][i] for i in range(3)],poses[cid][-1],intr,depth_tolerance=.05,pixel_radius=0)
    for branch in BRANCHES:
     for protocol in scale_protocols_for_branch(branch):
      if protocol=='memory': candidate_scale=calibrations[cid]['candidate_scale']
      elif protocol=='fixed_baseline': candidate_scale=base_scale
      else: candidate_scale=None
      candidate_views,meta=recover_branch_points_by_view(caches[cid],poses[cid],obs,branch,depth_scale=candidate_scale)
      states=build_disturbance_states_from_sampled_h0(shared_h0[branch],candidate_views,args.quota,args.reconstruction_sample_seed)
      candidate_sample=states['H2'][states['H0'].shape[0]:]
      runs.append({"candidate":cid,"branch":branch,"protocol":protocol,"target_size_limit":target_size,"target_sample_seed":target_seed,"reconstruction_sample_seed":args.reconstruction_sample_seed,"target_source_count":source_count,"target_sample_count":int(target.shape[0]),"target_was_subsampled":source_count>target.shape[0],"per_view_quota":args.quota,"baseline_scale":base_scale,"candidate_scale":meta['depth_scale'],"per_cache_candidate_scale":meta['per_cache_depth_scale'],"metrics":state_metrics(target,states,masks.observed,masks.novel),"candidate_view_outlier_ratio":outliers(candidate_sample,target),"h0_fingerprint":h0_fingerprints[branch]})
 summary={}; rankings={}
 for branch in ('C','D'):
  summary[branch]={}; rankings[branch]={}
  for cid in cands:
   rows=[r for r in runs if r['branch']==branch and r['protocol']=='memory' and r['target_size_limit']==12000 and r['candidate']==cid]
   summary[branch][cid]={region:{t:{effect:stats([r['metrics']['effects'][region][t][effect] for r in rows]) for effect in ('candidate_addition','history_recompute','joint_net')} for t in ('0.05','0.1','0.2')} for region in ('novel','observed')}
  seed_orders=[]
  for seed in range(5):
   vals=[next(r for r in runs if r['branch']==branch and r['protocol']=='memory' and r['target_size_limit']==12000 and r['target_sample_seed']==seed and r['candidate']==c)['metrics']['effects']['novel']['0.1']['candidate_addition'] for c in cands]
   seed_orders.append(rank(vals,cands))
  mean_vals=[summary[branch][c]['novel']['0.1']['candidate_addition']['mean'] for c in cands]
  dense_vals=[next(r for r in runs if r['branch']==branch and r['protocol']=='memory' and r['target_size_limit']==50000 and r['candidate']==c)['metrics']['effects']['novel']['0.1']['candidate_addition'] for c in cands]
  uncertainty_pairs=[]
  for i,first in enumerate(cands):
   for second in cands[i+1:]:
    a=summary[branch][first]['novel']['0.1']['candidate_addition']; b=summary[branch][second]['novel']['0.1']['candidate_addition']
    if uncertainty_intervals_overlap(a['mean'],a['std'],b['mean'],b['std']): uncertainty_pairs.append([first,second])
  rankings[branch]={"seed_orders_12k":seed_orders,"aggregate_order_12k":rank(mean_vals,cands),"order_50k":rank(dense_vals,cands),"spearman_12k_vs_50k":spearman_rank_correlation(mean_vals,dense_vals),"uncertainty_ties_12k_10cm":uncertainty_pairs}
 rankings['C_vs_D_12k']=spearman_rank_correlation([summary['C'][c]['novel']['0.1']['candidate_addition']['mean'] for c in cands],[summary['D'][c]['novel']['0.1']['candidate_addition']['mean'] for c in cands])
 result={"stage":"C7.1","status":"complete","did_run_vggt":False,"reused_cache_commit":"25490b5","base_scale":base_scale,"calibrations":calibrations,"shared_h0":h0_fingerprints,"run_count":len(runs),"memory_summary":summary,"rankings":rankings,"runs":runs}
 docs=Path(cfg['docs-output-dir']).parent/'scannet_scene0000_00_stage_c7_1'; docs.mkdir(parents=True,exist_ok=True); (docs/'stage_c7_1_corrected_scale_summary.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({k:v for k,v in result.items() if k not in ('runs','memory_summary')},indent=2))
if __name__=='__main__': main()
