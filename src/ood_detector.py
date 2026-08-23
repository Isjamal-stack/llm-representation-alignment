"""
Two ways to check whether stitched activations are out-of-distribution
relative to the real target model's own activations at that layer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def simple_distribution_distance(stitched_feat: torch.Tensor, real_feat: torch.Tensor) -> float:
    """
    Fast, training-free proxy for OOD-ness. Compares the mean-pooled
    stitched activation against the real target activation's own
    per-channel mean/std at the same layer, as a normalized distance:

        distance = || (mean(stitched) - mean(real)) / std(real) ||_2 / sqrt(C)

    This is a much lighter substitute for the full energy-based detector
    below (which needs a separate auxiliary out-of-distribution dataset
    and considerably more compute) -- good enough to answer "is TLT pushing
    activations further from the real distribution than FFM/LFM does" as a
    first pass. Larger distance = further from where real activations for
    this layer actually live = more likely OOD. Use the energy-based
    detector for a more rigorous follow-up once an auxiliary OOD dataset
    is chosen.
    """
    real_mean = real_feat.mean(dim=(0, 1))
    real_std = real_feat.std(dim=(0, 1)).clamp_min(1e-6)
    stitched_mean = stitched_feat.mean(dim=(0, 1))
    z = (stitched_mean - real_mean) / real_std
    return z.norm(p=2).item() / (real_feat.shape[-1] ** 0.5)


class SimpleOODHead(nn.Module):
    """A small classifier trained on pooled layer activations, fine-tuned
    to separate energy scores of in-distribution vs. auxiliary-OOD inputs."""
    def __init__(self, in_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.pool = lambda x: x.mean(dim=1)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, feat):
        return self.net(self.pool(feat))

    def energy(self, feat):
        logits = self.forward(feat)
        return -torch.logsumexp(logits, dim=-1)


def energy_margin_loss(e_in, e_out, m_in=-25.0, m_out=-7.0):
    return (F.relu(e_in - m_in) ** 2).mean() + (F.relu(m_out - e_out) ** 2).mean()


def pretrain_classification(head: SimpleOODHead, feats, labels, epochs=10, lr=1e-3):
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        logits = head(feats)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
    return head


def finetune_energy(head: SimpleOODHead, in_feats, in_labels, out_feats,
                     epochs=10, lr=1e-4, lam=0.1, m_in=-25.0, m_out=-7.0):
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        logits_in = head(in_feats)
        ce = F.cross_entropy(logits_in, in_labels)
        e_in = head.energy(in_feats)
        e_out = head.energy(out_feats)
        loss = ce + lam * energy_margin_loss(e_in, e_out, m_in, m_out)
        loss.backward()
        opt.step()
    return head


@torch.no_grad()
def ood_auroc(head: SimpleOODHead, real_target_feats, stitched_feats):
    e_real = head.energy(real_target_feats).cpu().numpy()
    e_stitched = head.energy(stitched_feats).cpu().numpy()
    scores = list(e_real) + list(e_stitched)
    labels = [0] * len(e_real) + [1] * len(e_stitched)
    return roc_auc_score(labels, scores)
