#!/usr/bin/env python3
"""Stage C6 append-only and joint-recompute disturbance audit."""
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
from viewtoken.oracle import (PinholeIntrinsics, build_memory_id, build_visibility_masks, build_disturbance_states, load_gt_poses_for_view_ids, load_point_cloud, load_pose_matrix, recover_branch_points_by_view, spearman_rank_correlation, state_metrics)
from viewtoken.oracle.metrics import nearest_neighbor_squared_distances, sample_points

BRANCHES=("B","C","D")
PROTOCOLS=("per_cache","fixed_baseline")

def parse_args():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=Path("configs/oracle_stage_c5_v4_smoke.yaml")); p.add_argument("--quota",type=int,default=3000); return p.parse_args()

def outliers(points, target, thresholds=(.1,.2)):
 d=nearest_neighbor_squared_distances(points,target).sqrt(); return {f"{t:g}":float((d>t).float().mean()) for t in thresholds}

def stats(values):
 a=np.asarray(values,dtype=float); return {"mean":float(a.mean()),"std":float(a.std()),"min":float(a.min()),"max":float(a.max()),"positive_fraction":float((a>0).mean())}

def main():
 args=parse_args(); cfg=load_config(args.config); observed=[Path(x).stem for x in cfg["observed-views"]]; candidates=[Path(x).stem for x in cfg["candidate-views"]]
 root=Path(cfg["output-dir"])/"reconstructions"; memory=build_memory_id(cfg["scene-id"],observed); pose_dir=Path(cfg["posed-image-dir"]); observed_paths=[Path(x) for x in cfg["observed-views"]]
 baseline=load_expected_cache(root/memory,observed_paths,cfg); base_poses=load_gt_poses_for_view_ids(observed,pose_dir)
 base_native={b:recover_branch_points_by_view(baseline,base_poses,observed,b) for b in BRANCHES}; baseline_scale=base_native["C"][1]["per_cache_depth_scale"]
 source=load_point_cloud(Path(cfg["target-points"]),point_stride=int(cfg.get("point-stride",6)))
 matrix=torch.tensor(np.loadtxt(cfg["intrinsics"],dtype=np.float32))[:3,:3]
 with Image.open(cfg["observed-views"][0]) as im: intr=PinholeIntrinsics.from_matrix(matrix,im.width,im.height)
 runs=[]
 for target_size,seeds in ((12000,range(5)),(50000,range(1))):
  for seed in seeds:
   target=sample_points(source,target_size,seed=seed,method="hash")
   for cid,cpath in zip(candidates,map(Path,cfg["candidate-views"])):
    ids=observed+[cid]; poses=load_gt_poses_for_view_ids(ids,pose_dir); cache=load_expected_cache(root/f"{memory}__plus__{cid}",observed_paths+[cpath],cfg)
    masks=build_visibility_masks(target,[poses[i] for i in range(3)],poses[-1],intr,depth_tolerance=.05,pixel_radius=0)
    for branch in BRANCHES:
     for protocol in PROTOCOLS:
      fixed=baseline_scale if protocol=="fixed_baseline" else None
      bviews,bmeta=recover_branch_points_by_view(baseline,base_poses,observed,branch,depth_scale=fixed)
      cviews,cmeta=recover_branch_points_by_view(cache,poses,observed,branch,depth_scale=fixed)
      states=build_disturbance_states(bviews,cviews,args.quota,seed)
      metric=state_metrics(target,states,masks.observed,masks.novel)
      sampled_candidate=states["H2"][states["H0"].shape[0]:]
      runs.append({"candidate":cid,"branch":branch,"scale_protocol":protocol,"target_size":target_size,"seed":seed,"quota":args.quota,"baseline_scale":bmeta["depth_scale"],"candidate_scale":cmeta["depth_scale"],"per_cache_candidate_scale":cmeta["per_cache_depth_scale"],"metrics":metric,"outlier_ratio":{"H0_observed_views":outliers(states["H0"],target),"H1_observed_views":outliers(states["H1"],target),"candidate_view":outliers(sampled_candidate,target)}})
 summary={}
 for branch in BRANCHES:
  summary[branch]={}
  for protocol in PROTOCOLS:
   rows=[r for r in runs if r["branch"]==branch and r["scale_protocol"]==protocol and r["target_size"]==12000]
   summary[branch][protocol]={}
   for cid in candidates:
    cr=[r for r in rows if r["candidate"]==cid]; summary[branch][protocol][cid]={}
    summary[branch][protocol][cid]["effects"]={region:{t:{effect:stats([x["metrics"]["effects"][region][t][effect] for x in cr]) for effect in ("candidate_addition","history_recompute","joint_net","observed_recompute")} for t in ("0.05","0.1","0.2")} for region in ("novel","observed")}
 rankings={}
 for branch in BRANCHES:
  a=[]; b=[]
  for cid in candidates:
   a.append(summary[branch]["per_cache"][cid]["effects"]["novel"]["0.1"]["candidate_addition"]["mean"]); b.append(summary[branch]["fixed_baseline"][cid]["effects"]["novel"]["0.1"]["candidate_addition"]["mean"])
  rankings[branch]={"spearman_per_cache_vs_fixed_baseline":spearman_rank_correlation(a,b),"per_cache_order":[x for _,x in sorted(zip(a,candidates),reverse=True)],"fixed_baseline_order":[x for _,x in sorted(zip(b,candidates),reverse=True)]}
 result={"stage":"C6","status":"complete","did_run_vggt":False,"reused_cache_commit":"25490b5","quota":args.quota,"run_count":len(runs),"baseline_scale":baseline_scale,"runs":runs,"sampling_summary":summary,"rankings":rankings}
 docs=Path(cfg["docs-output-dir"]).parent/"scannet_scene0000_00_stage_c6"; docs.mkdir(parents=True,exist_ok=True); (docs/"stage_c6_disturbance_summary.json").write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps({k:v for k,v in result.items() if k not in {"runs","sampling_summary"}},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
