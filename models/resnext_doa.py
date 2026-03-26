"""
resnext_doa.py
--------------
Modified ResNeXt-50 for multi-label DOA estimation.

Architecture (follows paper Table 1):
  Input  : (B, 2, H, W)    -- 2-channel input (real/imag)
           H=P, W=T  for raw input  (12, 16) or (12, 32)
           H=P, W=P  for cov input  (12, 12)
  Conv1  : 7x7, 64 filters, stride 2, padding 3
  Pool   : 3x3 max-pool, stride 2
  Stage1 : 3 x ResNeXt bottleneck block  (64-d,  32 groups, width=4)
  Stage2 : 4 x ResNeXt bottleneck block  (128-d, 32 groups, width=4), stride 2
  Stage3 : 6 x ResNeXt bottleneck block  (256-d, 32 groups, width=4), stride 2
  Stage4 : 3 x ResNeXt bottleneck block  (512-d, 32 groups, width=4), stride 2
  GAP    : Global Average Pooling -> (B, 2048)
  FC     : Linear(2048, 121) + Sigmoid

Key changes vs standard ResNeXt-50:
  1. Conv1 in-channels changed from 3 -> 2
  2. Final FC changed from Linear(2048, 1000) -> Linear(2048, 121)
  3. Sigmoid activation added (BCELoss training)
"""

import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights


class ResNeXtDOA(nn.Module):
    """
    Modified ResNeXt-50 (32x4d) for multi-label DOA estimation.

    Parameters
    ----------
    num_classes : int   Number of DOA classes (default 121).
    input_type  : str   'raw' or 'cov' -- only affects how we describe input.
    pretrained  : bool  Load ImageNet weights for all layers except Conv1 & FC.
    """

    def __init__(
        self,
        num_classes: int = 121,
        input_type: str = 'raw',
        pretrained: bool = True,
    ):
        super().__init__()
        self.input_type  = input_type
        self.num_classes = num_classes

        # ── Load standard ResNeXt-50 ───────────────────────────────────────────
        if pretrained:
            backbone = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        else:
            backbone = resnext50_32x4d(weights=None)

        # ── Modify Conv1: 3-channel -> 2-channel (real/imag) ──────────────────
        old_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False,
        )
        if pretrained:
            # Average the first two RGB channels to initialise the 2-channel conv
            # so we can still benefit from pretrained features.
            with torch.no_grad():
                new_conv1.weight.copy_(
                    old_conv1.weight[:, :2, :, :] +
                    old_conv1.weight[:, 2:3, :, :].repeat(1, 2, 1, 1) * 0.5
                )
        backbone.conv1 = new_conv1

        # ── Replace FC head: 1000 -> 121 classes ──────────────────────────────
        in_features = backbone.fc.in_features   # 2048
        backbone.fc  = nn.Linear(in_features, num_classes)

        # ── Store backbone sub-modules ─────────────────────────────────────────
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1   # Stage 1 (3 blocks)
        self.layer2  = backbone.layer2   # Stage 2 (4 blocks)
        self.layer3  = backbone.layer3   # Stage 3 (6 blocks)
        self.layer4  = backbone.layer4   # Stage 4 (3 blocks)
        self.avgpool = backbone.avgpool  # Global Average Pooling
        self.fc      = backbone.fc
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 2, H, W)

        Returns
        -------
        out : torch.Tensor, shape (B, 121)   values in (0, 1)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)   # (B, 2048)
        x = self.fc(x)
        x = self.sigmoid(x)        # multi-label probabilities

        return x

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick shape test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    for itype, shape in [('raw', (2, 12, 16)),
                          ('raw', (2, 12, 32)),
                          ('cov', (2, 12, 12))]:
        model = ResNeXtDOA(num_classes=121, input_type=itype,
                           pretrained=False).to(device)
        x = torch.randn(4, *shape, device=device)
        out = model(x)
        print(f'input_type={itype}  x.shape={tuple(x.shape)}  '
              f'out.shape={tuple(out.shape)}  '
              f'params={model.count_parameters():,}')
        assert out.shape == (4, 121), f'Shape mismatch: {out.shape}'

    print('All shape tests passed.')
