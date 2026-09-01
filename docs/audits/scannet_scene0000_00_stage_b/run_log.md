# Stage B Camera-Anchor Alignment Audit

Status: `blocked_missing_pose_enc`

This Stage-B audit did not run VGGT and did not create new reconstructions. It attempted to reuse the existing audit20 cached reconstructions, but those caches do not contain `pose_enc.pt`, which is required to decode VGGT-predicted camera centers.

## Inputs

- Audit records: `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/outputs/oracle_gain/scannet_scene0000_00_audit20/oracle_gain.jsonl`
- Target points: `/mnt/datasets/scannet-processed/26-06-21-1/ScanNet_processed/points/scene0000_00.bin`
- GT pose dir: `/mnt/datasets/scannet-processed/26-06-21-1/ScanNet_processed/posed_images/scene0000_00`
- Attempted records: `21`
- Missing reconstruction pose files: `22`

## Missing Files

- `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/outputs/oracle_gain/scannet_scene0000_00_audit20/reconstructions/23a55d429693f934/pose_enc.pt`
- `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/outputs/oracle_gain/scannet_scene0000_00_audit20/reconstructions/23a55d429693f934__plus__00010/pose_enc.pt`
- `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/outputs/oracle_gain/scannet_scene0000_00_audit20/reconstructions/23a55d429693f934__plus__00019/pose_enc.pt`
- `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/outputs/oracle_gain/scannet_scene0000_00_audit20/reconstructions/23a55d429693f934__plus__00018/pose_enc.pt`
- `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/outputs/oracle_gain/scannet_scene0000_00_audit20/reconstructions/23a55d429693f934__plus__00009/pose_enc.pt`
- `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/outputs/oracle_gain/scannet_scene0000_00_audit20/reconstructions/23a55d429693f934__plus__00032/pose_enc.pt`
- `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/outputs/oracle_gain/scannet_scene0000_00_audit20/reconstructions/23a55d429693f934__plus__00025/pose_enc.pt`
- `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/outputs/oracle_gain/scannet_scene0000_00_audit20/reconstructions/23a55d429693f934__plus__00050/pose_enc.pt`
- `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/outputs/oracle_gain/scannet_scene0000_00_audit20/reconstructions/23a55d429693f934__plus__00075/pose_enc.pt`
- `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/outputs/oracle_gain/scannet_scene0000_00_audit20/reconstructions/23a55d429693f934__plus__00100/pose_enc.pt`
- `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/outputs/oracle_gain/scannet_scene0000_00_audit20/reconstructions/23a55d429693f934__plus__00125/pose_enc.pt`
- `/mnt/volumes/ad-base-vla-vol-ga/mayongjia/ad-base-vla-vol-ga/group/cjy/ViewToken/outputs/oracle_gain/scannet_scene0000_00_audit20/reconstructions/23a55d429693f934__plus__00150/pose_enc.pt`
- ... 10 more

## Interpretation

Stage B requires each baseline and observed-plus-candidate reconstruction cache to store VGGT `pose_enc.pt`. The existing audit20 cache predates that requirement and only stores `points.pt`, `confidence.pt`, and `metadata.json`. Under the current tips, the correct behavior is to stop here rather than rerun VGGT silently.

Future reconstruction caches now write `pose_enc.pt` and use cache schema `oracle-reconstruction-v3`; after the user approves regenerating caches, this script can compute camera-anchored metrics without using candidate cameras as alignment anchors.
