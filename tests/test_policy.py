import torch

from taxi_rl.policy import N_GLOBAL, SLOT_F, ZONE_F, ScoringExtractor


def _obs(b=4, s=5, z=8):
    torch.manual_seed(0)
    g = torch.randn(b, N_GLOBAL)
    slots = torch.randn(b, s, SLOT_F)
    slots[:, :, 0] = 1.0  # all slots valid
    zones = torch.randn(b, z, ZONE_F)
    return g, slots, zones


def _flat(g, slots, zones):
    return torch.cat([g, slots.flatten(1), zones.flatten(1)], 1)


def test_slot_permutation_equivariance():
    net = ScoringExtractor(5, 8)
    g, slots, zones = _obs()
    perm = torch.tensor([3, 0, 4, 1, 2])
    a = net.forward_actor(_flat(g, slots, zones))
    b = net.forward_actor(_flat(g, slots[:, perm], zones))
    assert torch.allclose(a[:, :5][:, perm], b[:, :5], atol=1e-5)  # rider logits follow the riders
    assert torch.allclose(a[:, 5:], b[:, 5:], atol=1e-5)  # zone + wait logits unchanged


def test_critic_is_permutation_invariant():
    net = ScoringExtractor(5, 8)
    g, slots, zones = _obs()
    perm = torch.tensor([4, 3, 2, 1, 0])
    v1 = net.forward_critic(_flat(g, slots, zones))
    v2 = net.forward_critic(_flat(g, slots[:, perm], zones))
    assert torch.allclose(v1, v2, atol=1e-5)
