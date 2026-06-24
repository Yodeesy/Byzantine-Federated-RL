"""Secure Aggregation via pairwise pseudorandom masking.

Implements a simplified version of the protocol by Bonawitz et al. (CCS 2017).
Each pair of workers (i, j) shares a secret key s_ij. Worker i uploads a
masked gradient:

    g̃_i = g_i + Σ_{j>i} PRNG(s_ij) - Σ_{j<i} PRNG(s_ji)

The server sums all masked gradients; the pairwise masks cancel exactly,
yielding the same result as plaintext FedAvg.

Security property:
    The server only observes masked gradients g̃_i and cannot recover any
    individual plaintext gradient g_i.

Cost for Byzantine robustness:
    FedPG-BR relies on pairwise L2 distances ||g_i - g_j|| to filter outliers.
    These distances cannot be computed from the masked gradients, so Byzantine
    filtering is completely disabled under SecAgg.

Note (simplification for course project):
    In a real deployment, shared keys are established via Diffie-Hellman key
    exchange directly between workers. Here we simulate DH key agreement on
    the server side for demonstration purposes.
"""

import torch
from typing import List, Tuple


class SecureAggregationProtocol:
    """Simplified secure aggregation protocol with pairwise PRNG masking.

    Attributes:
        num_workers: Number of participating workers.
        grad_dim: Flattened gradient dimension.
        device: Torch device.
    """

    def __init__(self, num_workers: int, grad_dim: int, device: torch.device):
        """Initialize the protocol and simulate DH key agreement.

        Args:
            num_workers: Number of participating workers.
            grad_dim: Dimension of the flattened gradient vector.
            device: Torch device for tensor operations.
        """
        self.num_workers = num_workers
        self.grad_dim = grad_dim
        self.device = device

        # Simulate Diffie-Hellman key exchange: s_keys[i][j] == s_keys[j][i].
        # In a real deployment, these keys are established directly between
        # workers via DH and are invisible to the server.
        self._keys = [[0] * num_workers for _ in range(num_workers)]
        for i in range(num_workers):
            for j in range(i + 1, num_workers):
                seed = int(torch.randint(1, 2 ** 31 - 1, (1,)).item())
                self._keys[i][j] = seed
                self._keys[j][i] = seed

        print(f"[SecAgg] {num_workers} workers, grad_dim={grad_dim}")

    def _prng(self, seed: int, dim: int) -> torch.Tensor:
        """Deterministic pseudorandom number generator.

        Produces a reproducible mask vector from a seed. The same seed always
        yields the same output — this is the cryptographic basis for mask
        cancellation during server-side aggregation.

        Args:
            seed: Integer seed for the PRNG.
            dim: Output vector dimension.

        Returns:
            Random tensor of shape [dim].
        """
        gen = torch.Generator()
        gen.manual_seed(seed % (2 ** 31 - 1))
        return torch.randn(dim, generator=gen).to(self.device)

    def encrypt_gradient(self, worker_id: int,
                         flat_grad: torch.Tensor) -> torch.Tensor:
        """Apply pairwise masks to a worker's gradient (encryption).

        Implements: g̃_i = g_i + Σ_{j>i} PRNG(s_ij) - Σ_{j<i} PRNG(s_ji).

        Args:
            worker_id: Zero-indexed worker identifier.
            flat_grad: Plaintext gradient vector, shape [grad_dim].

        Returns:
            Masked gradient vector, shape [grad_dim].
        """
        i = worker_id
        mask = torch.zeros_like(flat_grad)
        for j in range(self.num_workers):
            if j == i:
                continue
            elif j > i:
                mask += self._prng(self._keys[i][j], self.grad_dim)
            else:
                mask -= self._prng(self._keys[j][i], self.grad_dim)
        return flat_grad + mask

    def verify_cancellation(self,
                            original: List[torch.Tensor],
                            masked: List[torch.Tensor]) -> float:
        """Verify that pairwise masks cancel correctly during aggregation.

        Checks that sum(masked) ≈ sum(original) within floating-point
        tolerance.

        Args:
            original: List of plaintext gradient vectors.
            masked: List of masked gradient vectors.

        Returns:
            Maximum element-wise absolute error.
        """
        sum_orig = torch.stack(original).sum(dim=0)
        sum_mask = torch.stack(masked).sum(dim=0)
        err = (sum_orig - sum_mask).abs().max().item()
        print(f"[SecAgg] mask cancel check: max_err={err:.2e} "
              f"{'[PASS]' if err < 1e-4 else '[FAIL]'}")
        return err

    def aggregate_masked(self,
                         masked_grads: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate masked gradients (server-side).

        Equivalent to computing mean(original_grads) without ever seeing
        individual plaintext gradients.

        Args:
            masked_grads: List of masked gradient vectors.

        Returns:
            Averaged gradient vector, shape [grad_dim].
        """
        return torch.stack(masked_grads).sum(dim=0) / self.num_workers


def flatten_gradient_list(gradient):
    """Flatten a list-of-list-of-tensors gradient representation.

    Args:
        gradient: gradient[i][p] = tensor for worker i, parameter p.

    Returns:
        Tuple of (flat_tensor [num_workers, total_dim], list of shapes).
    """
    shapes = [p.shape for p in gradient[0]]
    flat = [torch.cat([p.data.view(-1) for p in wg]) for wg in gradient]
    return torch.stack(flat), shapes


def unflatten_gradient(flat_vec, shapes):
    """Reshape a flat gradient vector back to per-parameter tensors.

    Args:
        flat_vec: Flattened vector, shape [total_dim].
        shapes: List of per-parameter tensor shapes.

    Returns:
        List of gradient tensors with the given shapes.
    """
    grads, offset = [], 0
    for s in shapes:
        n = s.numel()
        grads.append(flat_vec[offset:offset + n].view(s).clone())
        offset += n
    return grads
