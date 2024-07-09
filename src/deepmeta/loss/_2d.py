"""loss functions for 2d models"""
from torch import nn, Tensor
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from itertools import filterfalse as ifilterfalse

"""
#########################
## UNet Loss Function
#########################
Implemented and modified using:
https://github.com/hayashimasa/UNet-PyTorch/blob/main/loss.py
"""
class Weighted_Cross_Entropy_Loss(torch.nn.Module):
    def __init__(self):
        super(Weighted_Cross_Entropy_Loss, self).__init__()

    def forward(self, pred, target, weights=torch.tensor([1,15]).to('cuda')):
        logp = F.log_softmax(pred, dim=1)
        logp = torch.gather(logp, 1, target.unsqueeze(1))
        weighted_logp = (logp * weights.view(1, -1, 1, 1))
        weighted_loss = weighted_logp.sum(1) / weights.view(1, -1).sum(1)
        weighted_loss = -weighted_loss.mean()
        return weighted_loss

"""
#########################
## UNet3p Loss Function
#########################
Implemented and modified using: https://github.com/dmMaze/UNet3Plus-pytorch
"""
def binary_iou_loss(pred, target):
    Iand = torch.sum(pred * target, dim=1)
    Ior = torch.sum(pred, dim=1) + torch.sum(target, dim=1) - Iand
    IoU = 1 - Iand.sum() / Ior.sum()
    return IoU.sum()

class IoULoss(nn.Module):
    def __init__(self, process_input=True) -> None:
        super().__init__()
        self.process_input=process_input

    def forward(self, pred, target):
        num_classes = pred.shape[1]

        if self.process_input:
            pred = F.softmax(pred, dim=1)
            target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        total_loss = 0
        for i in range(num_classes):
            loss = binary_iou_loss(pred[:, i], target[:, i])
            total_loss += loss
        return total_loss / num_classes

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class U3PLloss(nn.Module):
    def __init__(self, loss_type='focal', aux_weight=0.4, process_input=True):
        super().__init__()
        self.aux_weight = aux_weight
        self.focal_loss = FocalLoss(ignore_index=255, size_average=True)
        if loss_type == 'u3p':
            self.iou_loss = IoULoss(process_input=not process_input)
            self.ms_ssim_loss = MS_SSIMLoss(process_input=not process_input)
        elif loss_type != 'focal':
            raise ValueError(f'Unknown loss type: {loss_type}')
        self.loss_type = loss_type
        self.process_input = process_input

    def forward(self, preds, targets):
        if not isinstance(preds, dict):
            preds = {'final_pred': preds}
        if self.loss_type == 'focal':
            return self._forward_focal(preds, targets)
        elif self.loss_type == 'u3p':
            return self._forward_u3p(preds, targets)

    def _forward_focal(self, preds, targets):
        loss_dict = {}
        loss = self.focal_loss(preds['final_pred'], targets)
        loss_dict['head_focal_loss'] = loss.detach().item()     # for logging
        num_aux, aux_loss = 0, 0.

        for key in preds:
            if 'aux' in key:
                num_aux += 1
                aux_loss += self.focal_loss(preds[key], targets)
        if num_aux > 0:
            aux_loss = aux_loss / num_aux * self.aux_weight
            loss_dict['aux_focal_loss'] = aux_loss.detach().item()
            loss += aux_loss
            loss_dict['total_loss'] = loss.detach().item()
        return loss

    def onehot_softmax(self, pred, target: torch.Tensor, process_target=True):
        _, num_classes, h, w = pred.shape
        pred = F.softmax(pred, dim=1)
        if process_target:
            target = torch.clamp(target, 0, num_classes)
            target = F.one_hot(target, num_classes=num_classes+1)[..., :num_classes].permute(0, 3, 1, 2).contiguous().to(pred.dtype)
        return pred, target

    def _forward_u3p(self, preds, targets):
        loss, loss_dict = self._forward_focal(preds, targets)
        if self.process_input:
            final_pred, targets = self.onehot_softmax(preds['final_pred'], targets)
        iou_loss = self.iou_loss(final_pred, targets)
        msssim_loss = self.ms_ssim_loss(final_pred, targets)
        loss = loss + iou_loss + msssim_loss
        loss_dict['head_iou_loss'] = iou_loss.detach().item()
        loss_dict['head_msssim_loss'] = msssim_loss.detach().item()
        num_aux, aux_iou_loss, aux_msssim_loss = 0, 0., 0.
        for key in preds:
            if 'aux' in key:
                num_aux += 1
                if self.process_input:
                    preds[key], targets = self.onehot_softmax(preds[key],
                                                              targets, process_target=False)
                aux_iou_loss += self.iou_loss(preds[key], targets)
                aux_msssim_loss += self.ms_ssim_loss(preds[key], targets)
        if num_aux > 0:
            aux_iou_loss /= num_aux
            aux_msssim_loss /= num_aux
            loss_dict['aux_iou_loss'] = aux_iou_loss.detach().item()
            loss_dict['aux_msssim_loss'] = aux_msssim_loss.detach().item()
            loss += (aux_iou_loss + aux_msssim_loss) * self.aux_weight
            loss_dict['total_loss'] = loss.detach().item()
        return loss

class SSIMLoss(nn.Module):
    def __init__(self, win_size: int = 11, nonnegative: bool = True, process_input: bool = True):
        super(SSIMLoss, self).__init__()
        self.kernel = gaussian_kernel2d(win_size, 1)
        self.win_size = win_size
        self.nonnegative = nonnegative
        self.process_input = process_input

    def forward(self, pred: Tensor, target: Tensor):
        _, num_classes, h, w = pred.shape
        win_size = min(h, w, self.win_size)
        kernel = self.kernel if win_size == self.win_size else gaussian_kernel2d(win_size, 1)
        kernel = kernel.to(pred.dtype).to(pred.device)
        if self.process_input:
            pred = F.softmax(pred, dim=1)
            target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        loss = 0.
        for i in range(num_classes):
            ss, _ = ssim_index(pred[:, [i]], target[:, [i]], kernel, nonnegative=self.nonnegative)
            loss += 1. - ss.mean()
        return loss / num_classes

class MS_SSIMLoss(nn.Module):
    def __init__(self,
                 win_size: int = 11,
                 weights: Tensor = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]), 
                 nonnegative: bool = True,
                 process_input: bool = True):

        super(MS_SSIMLoss, self).__init__()
        self.kernel = gaussian_kernel2d(win_size, 1)
        self.weights = weights
        self.win_size = win_size
        self.nonnegative = nonnegative
        self.process_input = process_input

    def forward(self, pred: Tensor, target: Tensor):
        _, num_classes, h, w = pred.shape
        win_size = min(h, w, self.win_size)
        kernel = self.kernel if win_size == self.win_size else gaussian_kernel2d(win_size, 1)
        kernel = kernel.to(pred.dtype).to(pred.device)
        weights = self.weights.to(pred.dtype).to(pred.device)
        if self.process_input:
            pred = F.softmax(pred, dim=1)
            target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        loss = 0.
        for i in range(num_classes):
            ss = ms_ssim(pred[:, [i]], target[:, [i]], kernel, weights, nonnegative=self.nonnegative)
            loss += 1. - ss.mean()
        return loss / num_classes

def ms_ssim(
    x: Tensor,
    y: Tensor,
    kernel: Tensor,
    weights: Tensor,
    val_range: float = 1.,
    nonnegative: bool = True) -> Tensor:
    css = []
    kernel_size = kernel.shape[-1]
    m = weights.numel()
    for i in range(m):
        if i > 0:
            x = F.avg_pool2d(x, kernel_size=2, ceil_mode=True)
            y = F.avg_pool2d(y, kernel_size=2, ceil_mode=True)
            h, w = x.shape[-2:]
            if h < kernel_size or w < kernel_size:
                weights = weights[:i] / torch.sum(weights[:i])
                break
        ss, cs = ssim_index(
            x, y, kernel,
            channel_avg=False,
            val_range=val_range,
            nonnegative=nonnegative
        )
        css.append(cs if i + 1 < m else ss)
    msss = torch.stack(css, dim=-1) ** weights
    msss = msss.prod(dim=-1).mean(dim=-1)
    return msss

def ssim_index(img1: Tensor, 
               img2: Tensor, 
               kernel: Tensor,
               nonnegative: bool = True,
               channel_avg: bool = False,
               val_range: float = 1.):
    relu = nn.ReLU(inplace=True)
    assert img1.shape == img2.shape
    if len(img1.shape) > 3:
        channel = img1.shape[1]
    else:
        img1 = img1.unsqueeze(1)
        img2 = img2.unsqueeze(1)
        channel = 1
    _, channel, height, width = img1.shape
    if img1.dtype == torch.long:
        img1 = img1.float()
    if img2.dtype == torch.long:
        img2 = img2.float()
    L = val_range

    s = 1
    p = 0
    mean1 = F.conv2d(img1, kernel, padding=p, groups=channel, stride=s)
    mean2 = F.conv2d(img2, kernel, padding=p, groups=channel, stride=s)
    mean12 = mean1 * mean2
    mean1 = mean1.pow(2)
    mean2 = mean2.pow(2)

    # https://en.wikipedia.org/wiki/Variance#Definition
    var1 = F.conv2d(img1 ** 2, kernel, padding=p, groups=channel, stride=s) - mean1
    var2 = F.conv2d(img2 ** 2, kernel, padding=p, groups=channel, stride=s) - mean2

    # https://en.wikipedia.org/wiki/Covariance#Definition
    covar = F.conv2d(img1 * img2, kernel, padding=p, groups=channel, stride=s) - mean12
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2

    # https://en.wikipedia.org/wiki/Structural_similarity#Algorithm
    cs = (2. * covar + c2) / (var1 + var2 + c2)
    ss = (2. * mean12 + c1) / (mean1 + mean2 + c1) * cs


    if channel_avg:
        ss, cs = ss.flatten(1), cs.flatten(1)
    else:
        ss, cs = ss.flatten(2), cs.flatten(2)

    ss, cs = ss.mean(dim=-1), cs.mean(dim=-1)
    if nonnegative:
        ss, cs = relu(ss), relu(cs)
    return ss, cs

def gaussian_kernel(kernel_size: int, sigma: float):
    gauss = torch.arange(0, kernel_size) - kernel_size // 2
    gauss = torch.exp(-gauss**2 / (2*sigma**2))
    return gauss / gauss.sum()

def gaussian_kernel2d(kernel_size: int, channel: int = 1) -> Tensor:
    k = gaussian_kernel(kernel_size, 1.5)
    k = torch.einsum('i,j->ij', [k, k])
    return k.expand(channel, 1, kernel_size, kernel_size).contiguous()

"""
#########################
## DeepMeta Loss Function
#########################
Implemented and modified usin https://github.com/cbib/DeepMeta
"""
class FusionLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.4,
        gamma: float = 0.2,
        device: str = "cuda",
        custom_weights: list = [1.0,5.0, 15.0] 
    ) -> None:
        super(FusionLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=torch.tensor(custom_weights).to(device),
            label_smoothing=0.1, 
        )
        self.focal = torch.hub.load(
            "adeelh/pytorch-multi-class-focal-loss",
            model="FocalLoss",
            alpha=torch.tensor(custom_weights).to(device),
            gamma=2,
            reduction="mean",
            force_reload=False,
        )
        self.lovasz = LovaszLoss(per_image=True)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return (
            self.alpha * self.ce(y_pred, y_true)
            + self.beta * self.lovasz(y_pred, y_true)  # noqa
            + self.gamma * self.focal(y_pred, y_true)) # noqa  # noqa

class LovaszLoss(nn.Module):
    def __init__(
        self, classes: str = "present", per_image: bool = False, ignore= None
    ) -> None:
        super(LovaszLoss, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.per_image:
            loss = mean(
                self.lovasz_softmax_flat(
                    *flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), self.ignore),
                    classes=self.classes,
                )
                for prob, lab in zip(y_pred.float(), y_true.float())
            )
        else:
            loss = self.lovasz_softmax_flat(
                *flatten_probas(y_pred.float(), y_true.float(), self.ignore), classes=self.classes
            )
        return loss

    @staticmethod
    def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[:-1]
        return jaccard

    def lovasz_softmax_flat(
        self, probas: torch.Tensor, labels: torch.Tensor, classes: str = "present"
    ) -> torch.Tensor:
        if probas.numel() == 0:
            return probas * 0.0
        C = probas.size(1)
        losses = []
        class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
        for c in class_to_sum:
            fg = (labels == c).float()  # foreground for class c
            if classes == "present" and fg.sum() == 0:
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError("Sigmoid output possible only with 1 class")
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (Variable(fg) - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(
                torch.dot(errors_sorted, Variable(self.lovasz_grad(fg_sorted)))
            )
        return mean(losses)

def mean(elt_list, ignore_nan: bool = False, empty: int = 0) -> torch.Tensor:
    elt_list = iter(elt_list)
    if ignore_nan:
        elt_list = ifilterfalse(isnan, elt_list)
    try:
        n = 1
        acc = next(elt_list)
    except StopIteration as e:
        if empty == "raise":
            raise ValueError("Empty mean") from e
        return empty
    for n, v in enumerate(elt_list, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

def flatten_probas(
    probas: torch.Tensor, labels: torch.Tensor, ignore= None
):
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def isnan(x: torch.Tensor) -> torch.Tensor:
    return x != x
