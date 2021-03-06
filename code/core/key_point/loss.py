import torch


class NullLoss:
    """
    训练时使用
    """
    def __call__(self, x, *args):
        return x.mean()


class KeyPointBCELoss:
    """
    在关键点模型中使用
    """
    def __init__(self, max_dist=6):
        self.max_dist = max_dist

    def __call__(self, pred: torch.Tensor, dist: torch.Tensor, mask: torch.Tensor):
        dist = dist.to(pred.device)

        pred = pred[mask]
        dist = dist[mask]
        label = dist < self.max_dist
        label = label.to(pred.dtype)

        loss = torch.nn.BCEWithLogitsLoss(pos_weight=1 / label.mean())
        return loss(pred, label)
