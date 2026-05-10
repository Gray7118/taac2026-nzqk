import torch
import torch.nn as nn
from typing import NamedTuple, Dict

class ModelInput(NamedTuple):
    features: torch.Tensor
    seq_data: Dict[str, torch.Tensor]

class Model(nn.Module):
    def forward(self, inputs: ModelInput) -> torch.Tensor:
        print(f"Inside forward: features shape {inputs.features.shape}, seq_a shape {inputs.seq_data['seq_a'].shape} on {inputs.features.device}")
        return inputs.features.sum(dim=1)

model = Model()
dp_model = nn.DataParallel(model, device_ids=[0, 1]).cuda()

inputs = ModelInput(
    features=torch.randn(128, 10).cuda(),
    seq_data={'seq_a': torch.randn(128, 5, 20).cuda()}
)

out = dp_model(inputs)
print(f"Output shape: {out.shape}")
