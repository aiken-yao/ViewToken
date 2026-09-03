import unittest
import torch
from viewtoken.oracle import append_only_coverage_is_monotonic, build_disturbance_states

class C6DisturbanceTest(unittest.TestCase):
    def test_append_only_coverage_never_decreases(self):
        target=torch.tensor([[0.,0.,0.],[1.,0.,0.]])
        base=torch.tensor([[0.,0.,0.]])
        added=torch.tensor([[1.,0.,0.]])
        self.assertTrue(append_only_coverage_is_monotonic(target, base, added))

    def test_h2_preserves_h0_prefix_exactly(self):
        base=[torch.randn(8,3) for _ in range(3)]
        joint=[torch.randn(8,3) for _ in range(4)]
        states=build_disturbance_states(base,joint,quota=5,seed=2)
        self.assertTrue(torch.equal(states["H2"][:len(states["H0"])], states["H0"]))

if __name__ == "__main__": unittest.main()
