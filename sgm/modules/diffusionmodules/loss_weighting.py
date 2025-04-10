from abc import ABC, abstractmethod

import torch


class DiffusionLossWeighting(ABC):
    @abstractmethod
    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        pass


class UnitWeighting(DiffusionLossWeighting):
    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(sigma, device=sigma.device)


class EDMWeighting(DiffusionLossWeighting):
    def __init__(self, sigma_data: float = 0.5):
        self.sigma_data = sigma_data

    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2


class VWeighting(EDMWeighting):
    def __init__(self):
        super().__init__(sigma_data=1.0)


class EpsWeighting(DiffusionLossWeighting):
    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma**-2.0
    
class SevaWeighting(DiffusionLossWeighting):
    def __call__(self, sigma: torch.Tensor, mask, max_weight=5.0) -> torch.Tensor:
        bools = mask.to(torch.bool)
        batch_size, num_frames = bools.shape
        indices = torch.arange(num_frames, device=bools.device).unsqueeze(0).expand(batch_size, num_frames)
        weights = torch.full((batch_size, num_frames), max_weight, dtype=torch.float, device=bools.device)
        
        for b in range(batch_size):
            true_idx = indices[b][bools[b]]
            if len(true_idx) > 0:
                dists = torch.stack([torch.abs(indices[b] - t) for t in true_idx]).min(dim=0).values
                dists[bools[b]] = 0
                weights[b] = dists / dists.max() * max_weight
            else:
                weights[b] = max_weight

        return weights

