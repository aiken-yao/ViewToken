#!/usr/bin/env python3
"""Stage C7 observed-depth memory scale calibration and C6 replay."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.extract_vggt_features import load_config
from scripts.audit_stage_c5_v4_branches import load_expected_cache
from viewtoken.oracle import (PinholeIntrinsics,build_memory_id,build_visibility_masks,calibrate_candidate_depth_scale,build_disturbance_states,load_gt_poses_for_view_ids,load_point_cloud,recover_branch_points_by_view,state_metrics)
from viewtoken.oracle.metrics import sample_points

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,default=Path('configs/oracle_stage_c5_v4_smoke.yaml')); ap.add_argument('--quota',type=int,default=3000); args=ap.parse_args(); cfg=load_config(args.config)
 obs=[Path(x).stem for x in cfg['observed-views']]; cands=[Path(x).stem for x in cfg['candidate-views']]; root=Path(cfg['output-dir'])/'reconstructions'; mem=build_memory_id(cfg['scene-id'],obs); pose_dir=Path(cfg['posed-image-dir']); obs_paths=[Path(x) for x in cfg['observed-views']]
 base=load_expected_cache(root/mem,obs_paths,cfg); poses0=load_gt_poses_for_view_ids(obs,pose_dir); base_views={b:recover_branch_points_by_view(base,poses0,obs,b) for b in ('B','C','D')}; base_scale=base_views['C'][1]['per_cache_depth_scale']
 source=load_point_cloud(Path(cfg['target-points']),point_stride=int(cfg.get('point-stride',6))); matrix=torch.tensor(np.loadtxt(cfg['intrinsics'],dtype=np.float32))[:3,:3]
 with Image.open(cfg['observed-views'][0]) as im: intr=PinholeIntrinsics.from_matrix(matrix,im.width,im.height)
 calibrations={}; runs=[]
 for cid,cpath in zip(cands,map(Path,cfg['candidate-views'])):
  cache=load_expected_cache(root/f'{mem}__plus__{cid}',obs_paths+[cpath],cfg); cal={b:calibrate_candidate_depth_scale(base.depth[:3],cache.depth[:3],3,base_scale) for b in ('C','D')}; calibrations[cid]=cal
  if any(x['status']!='calibrated' for x in cal.values()): raise RuntimeError(f'blocked_inconsistent_observed_depth_scale: {cid}')
  ids=obs+[cid]; poses=load_gt_poses_for_view_ids(ids,pose_dir)
  for size,seeds in ((12000,range(5)),(50000,range(1))):
   target=sample_points(source,size,seed=0 if size==50000 else 0,method='hash'); masks=build_visibility_masks(target,[poses[i] for i in range(3)],poses[-1],intr,depth_tolerance=.05,pixel_radius=0)
   for seed in seeds:
    for branch in ('B','C','D'):
     for protocol in ('memory','per_cache','fixed_baseline'):
      if protocol=='memory' and branch in ('C','D'): scale=cal[branch]['candidate_scale']
      elif protocol=='fixed_baseline': scale=base_scale
      else: scale=None
      bv,bm=recover_branch_points_by_view(base,poses0,obs,branch,depth_scale=scale); cv,cm=recover_branch_points_by_view(cache,poses,obs,branch,depth_scale=scale)
      states=build_disturbance_states(bv,cv,args.quota,seed); runs.append({'candidate':cid,'branch':branch,'protocol':protocol,'target_size':size,'seed':seed,'metrics':state_metrics(target,states,masks.observed,masks.novel),'scales':{'baseline':bm['depth_scale'],'candidate':cm['depth_scale'],'per_cache_candidate':cm['per_cache_depth_scale']}})
 out={'stage':'C7','status':'complete','did_run_vggt':False,'reused_cache_commit':'25490b5','base_scale':base_scale,'calibrations':calibrations,'run_count':len(runs),'runs':runs}
 docs=Path(cfg['docs-output-dir']).parent/'scannet_scene0000_00_stage_c7'; docs.mkdir(parents=True,exist_ok=True); (docs/'stage_c7_scale_calibration_summary.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'stage':'C7','status':'complete','did_run_vggt':False,'run_count':len(runs),'base_scale':base_scale,'calibrations':calibrations},indent=2))
if __name__=='__main__': main()
