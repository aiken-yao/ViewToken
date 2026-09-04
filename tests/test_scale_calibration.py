import unittest
import torch
from viewtoken.oracle import calibrate_candidate_depth_scale
class ScaleCalibrationTest(unittest.TestCase):
 def test_constant_ratio_recovers_scale(self):
  b=torch.ones(3,2,2)*4; c=b/2
  r=calibrate_candidate_depth_scale(b,c,3,1.5)
  self.assertEqual(r["status"],"calibrated"); self.assertAlmostEqual(r["candidate_scale"],3.0)
 def test_filters_nonpositive_and_nan(self):
  b=torch.tensor([[[2.,0.],[float("nan"),4.]]]*3); c=torch.tensor([[[1.,1.],[1.,2.]]]*3)
  r=calibrate_candidate_depth_scale(b,c,3,1.)
  self.assertEqual(r["ratio_stats"]["count"],6)
 def test_inconsistent_view_blocks(self):
  b=torch.ones(3,2,2); c=torch.stack([b[0],b[1]*.5,b[2]*2])
  r=calibrate_candidate_depth_scale(b,c,3,1.)
  self.assertEqual(r["status"],"blocked_inconsistent_observed_depth_scale")
 def test_candidate_extra_view_is_ignored(self):
  b=torch.ones(3,2,2); c=torch.cat([b/2,torch.ones(1,2,2)*100])
  r=calibrate_candidate_depth_scale(b,c,3,1.)
  self.assertAlmostEqual(r["candidate_scale"],2.)
 def test_order_mismatch_fails_shape(self):
  with self.assertRaises(ValueError): calibrate_candidate_depth_scale(torch.ones(3,2,2),torch.ones(4,3,2),3,1.)
if __name__=="__main__": unittest.main()
