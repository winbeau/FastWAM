"""CPU checks of the region-attrib round (fovea.token_latent_mask / token_norms against the video patch embedding's order)."""

import torch
from einops import rearrange

from fastwam.casestudy import fovea as fv


def _tokens(z):  # [1, c, f, 14, 28] -> [f, 98, c * 2 * 2], the order of the video patch embedding (patch 1 x 2 x 2)
    return rearrange(z[0], "c f (h b) (w e) -> f (h w) (c b e)", b=2, e=2)


def test_token_latent_mask_replaces_exactly_the_masked_tokens():
    g = torch.Generator().manual_seed(3)
    z1, zs = torch.randn(1, 4, 2, 14, 28, generator=g), torch.randn(1, 4, 2, 14, 28, generator=g)
    mask = torch.zeros(2, 98, dtype=torch.bool)
    mask[0, [0, 6, 7, 50, 97]] = True  # frame 1: agentview corner, agentview row 0 end, wrist row 0 start, ..., last
    mask[1, [13, 84]] = True
    lat = fv.token_latent_mask(mask.numpy(), 7, 14, 2, 2)
    assert lat.shape == (1, 1, 2, 14, 28) and int(lat.sum()) == 4 * int(mask.sum())
    z = torch.where(lat, zs, z1)
    t, t1, ts = _tokens(z), _tokens(z1), _tokens(zs)
    assert torch.equal(t[mask], ts[mask]) and torch.equal(t[~mask], t1[~mask])
    assert torch.equal(torch.where(fv.token_latent_mask(torch.ones(2, 98, dtype=torch.bool), 7, 14, 2, 2), zs, z1), zs)


def test_token_norms_is_the_token_vector_norm():
    g = torch.Generator().manual_seed(4)
    d = torch.randn(1, 5, 2, 14, 28, generator=g)
    assert torch.allclose(fv.token_norms(d, 7, 14, 2, 2), _tokens(d).norm(dim=-1), atol=1e-5)
