"""Observed-depth memory anchored scale calibration."""
from __future__ import annotations
from typing import Any
import torch
from torch import Tensor

def _stats(values: Tensor) -> dict[str, Any]:
    v=values[torch.isfinite(values)]
    if v.numel()==0: return {"count":0,"median":None,"p10":None,"p90":None,"mad":None}
    med=v.median(); mad=(v-med).abs().median()
    return {"count":int(v.numel()),"median":float(med),"p10":float(torch.quantile(v,.1)),"p90":float(torch.quantile(v,.9)),"mad":float(mad)}

def calibrate_candidate_depth_scale(baseline_depth: Tensor, candidate_depth: Tensor, observed_count: int, baseline_scale: float, max_relative_mad: float = 0.25, max_view_median_relative_spread: float = 0.5) -> dict[str, Any]:
    b=torch.as_tensor(baseline_depth,dtype=torch.float32).cpu(); c=torch.as_tensor(candidate_depth,dtype=torch.float32).cpu()
    if b.ndim!=3 or c.ndim!=3 or b.shape[0]<observed_count or c.shape[0]<observed_count or b.shape[1:]!=c.shape[1:]: raise ValueError("baseline/candidate observed depth shape mismatch")
    ratios=[]; per_view=[]
    for i in range(observed_count):
        valid=torch.isfinite(b[i]) & torch.isfinite(c[i]) & (b[i]>0) & (c[i]>0)
        r=b[i][valid]/c[i][valid]
        st=_stats(r); st["view_index"]=i; st["valid_ratio"]=float(valid.float().mean()); per_view.append(st)
        if r.numel()==0: raise ValueError(f"observed view {i} has no valid depth ratios")
        ratios.append(r)
    medians=torch.tensor([x["median"] for x in per_view])
    global_values=torch.cat(ratios); global_stats=_stats(global_values)
    spread=float((medians.max()-medians.min())/medians.median().clamp_min(1e-12))
    rel_mad=float(global_stats["mad"]/abs(global_stats["median"])) if global_stats["median"] else float("inf")
    blocked=spread>max_view_median_relative_spread or rel_mad>max_relative_mad
    return {"status":"blocked_inconsistent_observed_depth_scale" if blocked else "calibrated","baseline_scale":float(baseline_scale),"ratio_stats":global_stats,"per_view":per_view,"median_spread_relative":spread,"relative_mad":rel_mad,"candidate_scale":None if blocked else float(baseline_scale*global_stats["median"]),"observed_count":observed_count}


def scale_protocols_for_branch(branch: str) -> tuple[str, ...]:
    branch = branch.upper()
    if branch == "B":
        return ("per_cache", "fixed_baseline")
    if branch in {"C", "D"}:
        return ("memory", "per_cache", "fixed_baseline")
    raise ValueError(f"unsupported branch: {branch}")


def uncertainty_intervals_overlap(first_mean: float, first_std: float, second_mean: float, second_std: float) -> bool:
    """Return true when one-standard-deviation gain intervals overlap."""
    return first_mean - first_std <= second_mean + second_std and second_mean - second_std <= first_mean + first_std
