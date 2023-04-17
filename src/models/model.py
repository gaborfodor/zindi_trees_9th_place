import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
import timm
import numpy as np

def create_timm_model(model_name, pretrained, in_chans) -> torch.nn.Module:
    model = timm.create_model(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="",
        drop_rate=0,
        in_chans=in_chans,
    )
    return model

class MAELoss(nn.modules.Module):
    """Calculate MAE Loss"""

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.loss(logits, targets)
        return loss

class MSELoss(nn.modules.Module):
    """Calculate MSE Loss"""

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.loss(logits, targets)
        return loss

class RMSELoss(nn.modules.Module):
    """Calculate RMSE Loss"""

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.loss(logits, targets)
        loss = torch.sqrt(loss)
        return loss

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()

        self.cfg = cfg
        self.n_classes = len(self.cfg.classes)
        self.backbone = self._create_backbone()
        if cfg.loss == 'mse':
            self.loss_fn = MSELoss()
        elif cfg.loss == 'mae':
            self.loss_fn = MAELoss()

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.head = torch.nn.Linear(self.backbone.num_features, self.n_classes)

    def _create_backbone(self) -> torch.nn.Module:
        return create_timm_model(
            model_name=self.cfg.backbone,
            pretrained=self.cfg.pretrained,
            in_chans=self.cfg.in_chans,
        )

    def forward(self, batch, calculate_loss=True):
        x = batch["input"]
        targets = batch["target"].float()
        x = self.backbone(x)
        x = self.pooling(x)[:, :, 0, 0]

        if self.cfg.drop_out > 0.0:
            x = F.dropout(x, p=self.cfg.drop_out, training=self.training)
        logits = self.head(x)

        outputs = {}
        if not self.training:
            outputs["logits"] = logits

        if calculate_loss:
            outputs["target"] = targets
            outputs["loss"] = self.loss_fn(logits, targets)
        return outputs
