""" building blocks, implemented and modified from https://github.com/cbib/DeepMeta/tree/master/src/models """
from typing import Any, List
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    A double convolutional block (conv -> bn -> relu -> conv -> bn -> relu)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop_r: float,
        conv: nn.Module = nn.Conv2d,
    ):
        """
        Initialize the DoubleConv block
        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param drop_r: dropout rate
        :type drop_r: float
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(drop_r),
            conv(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(drop_r),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class DoubleConv3d(nn.Module):
    """
    A double convolutional block (conv -> bn -> relu -> conv -> bn -> relu)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop_r: float,
        conv: nn.Module = nn.Conv3d,
    ):
        """
        Initialize the DoubleConv block
        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param drop_r: dropout rate
        :type drop_r: float
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout3d(drop_r),
            conv(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout3d(drop_r),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class Down_Block(nn.Module):
    """Downscaling double conv then maxpool"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
    ):
        """
        Initialize the Down_Block
        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param drop: dropout rate
        :type drop: float
        """
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, drop, conv=conv_l)
        self.down = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        c = self.conv(x)
        return c, self.down(c)

class Down_Block3d(nn.Module):
    """Downscaling double conv then maxpool"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop: float = 0.1,
        conv_l: nn.Module = nn.Conv3d,
    ):
        """
        Initialize the Down_Block
        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param drop: dropout rate
        :type drop: float
        """
        super().__init__()
        self.conv = DoubleConv3d(in_channels, out_channels, drop, conv=conv_l)
        self.down = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor):
        c = self.conv(x)
        return c, self.down(c)

class Bridge(nn.Module):
    """
    Bridge block (conv -> bn -> relu -> conv -> bn -> relu)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
    ):
        """
        Initialize the Bridge block

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param drop: dropout rate
        :type drop: float
        """
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, drop, conv=conv_l)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Bridge3d(nn.Module):
    """
    Bridge block (conv -> bn -> relu -> conv -> bn -> relu)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop: float = 0.1,
        conv_l: nn.Module = nn.Conv3d,
    ):
        """
        Initialize the Bridge block

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param drop: dropout rate
        :type drop: float
        """
        super().__init__()
        self.conv = DoubleConv3d(in_channels, out_channels, drop, conv=conv_l)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Up_Block(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
    ):
        """
        Initialize the Up_Block

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param drop: dropout rate
        :type drop: float
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2)
        )
        self.conv = DoubleConv(in_channels, out_channels, drop, conv=conv_l)

    def forward(self, x: torch.Tensor, conc: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x)
        x = torch.cat([conc, x1], dim=1)
        return self.conv(x)

class Up_Block3d(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop: float = 0.1,
        conv_l: nn.Module = nn.Conv3d,
    ):
        """
        Initialize the Up_Block

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param drop: dropout rate
        :type drop: float
        """
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)
        )
        self.conv = DoubleConv3d(in_channels, out_channels, drop, conv=conv_l)

    def forward(self, x: torch.Tensor, conc: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x)
        x = torch.cat([conc, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    Output convolutional block (conv -> bn -> relu -> conv -> bn -> sigmoid)
    """

    def __init__(
        self, in_channels: int, out_channels: int, conv_l: nn.Module = nn.Conv2d
    ):
        """
        Initialize the OutConv block

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        """
        super().__init__()
        self.conv = conv_l(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class OutConv3d(nn.Module):
    """
    Output convolutional block (conv -> bn -> relu -> conv -> bn -> sigmoid)
    """

    def __init__(
        self, in_channels: int, out_channels: int, conv_l: nn.Module = nn.Conv3d
    ):
        """
        Initialize the OutConv block

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        """
        super().__init__()
        self.conv = conv_l(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

"""
################################################################################
## Tweaks
################################################################################
"""

class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: Any = "same",
        bias: bool = False,
        stride: int = 1,
    ):
        super().__init__()
        padding = 1 if stride > 1 else "same"
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=bias,
            padding=padding,
            stride=stride,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias, padding="same"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class SeparableConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding= "same",
        bias: bool = False,
        stride: int = 1,
    ):
        super().__init__()
        padding = 1 if stride > 1 else "same"
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=bias,
            padding=padding,
            stride=stride,
        )
        self.pointwise = nn.Conv3d(
            in_channels, out_channels, kernel_size=1, bias=bias, padding="same"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Normalize_Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int):
        super().__init__()
        self.down = nn.MaxPool2d(kernel)
        self.norm = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.down(x))

class Normalize_Down3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int):
        super().__init__()
        self.down = nn.MaxPool3d(kernel)
        self.norm = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.down(x))

class Normalize_Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=kernel)
        self.norm = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.up(x))

class Normalize_Up3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int):
        super().__init__()
        self.up = nn.Upsample(mode = 'trilinear', scale_factor=kernel)
        self.norm = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.up(x))

class Concat_Block_3p(nn.Module):
    def __init__(
        self,
        kernels_down,
        filters_down,
        kernels_up,
        filters_up,
    ):
        super().__init__()
        self.norm_down = nn.ModuleList(
            [
                Normalize_Down(in_, 64, kernel)
                for (kernel, in_) in zip(kernels_down, filters_down)
            ]
        )
        self.norm_up = nn.ModuleList(
            [
                Normalize_Up(in_, 64, kernel)
                for (kernel, in_) in zip(kernels_up, filters_up)
            ]
        )

    def forward(self, down, up) -> torch.Tensor:
        res = [l(d) for d, l in zip(down, self.norm_down)]
        res.extend(l(u) for u, l in zip(up, self.norm_up))
        return torch.cat(res, dim=1)

class Concat_Block_3p3d(nn.Module):
    def __init__(
        self,
        kernels_down,
        filters_down,
        kernels_up,
        filters_up,
    ):
        super().__init__()
        self.norm_down = nn.ModuleList(
            [
                Normalize_Down3d(in_, 64, kernel)
                for (kernel, in_) in zip(kernels_down, filters_down)
            ]
        )
        self.norm_up = nn.ModuleList(
            [
                Normalize_Up3d(in_, 64, kernel)
                for (kernel, in_) in zip(kernels_up, filters_up)
            ]
        )

    def forward(self, down, up) -> torch.Tensor:
        res = [l(d) for d, l in zip(down, self.norm_down)]
        res.extend(l(u) for u, l in zip(up, self.norm_up))
        return torch.cat(res, dim=1)

class Concat_Block(nn.Module):
    def __init__(
        self,
        kernels_down: List,
        filters_down: List,
        kernels_up: List,
        filters_up: List,
    ):
        super().__init__()
        self.norm_down = nn.ModuleList(
            [
                Normalize_Down(in_, 32, kernel)
                for (kernel, in_) in zip(kernels_down, filters_down)
            ]
        )
        self.norm_up = nn.ModuleList(
            [
                Normalize_Up(in_, 32, kernel)
                for (kernel, in_) in zip(kernels_up, filters_up)
            ]
        )

    def forward(self, down: List[torch.Tensor], up: List[torch.Tensor]) -> torch.Tensor:
        res = [l(d) for d, l in zip(down, self.norm_down)]
        res.extend(l(u) for u, l in zip(up, self.norm_up))
        return torch.cat(res, dim=1)

class Concat_Block3d(nn.Module):
    def __init__(
        self,
        kernels_down,
        filters_down,
        kernels_up,
        filters_up,
    ):
        super().__init__()
        self.norm_down = nn.ModuleList(
            [
                Normalize_Down3d(in_, 32, kernel)
                for (kernel, in_) in zip(kernels_down, filters_down)
            ]
        )
        self.norm_up = nn.ModuleList(
            [
                Normalize_Up3d(in_, 32, kernel)
                for (kernel, in_) in zip(kernels_up, filters_up)
            ]
        )

    def forward(self, down, up) -> torch.Tensor:
        res = [l(d) for d, l in zip(down, self.norm_down)]
        res.extend(l(u) for u, l in zip(up, self.norm_up))
        return torch.cat(res, dim=1)

class Up_Block_3p(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
    ):
        """
        Initialize the Up_Block

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param drop: dropout rate
        :type drop: float
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, 64, kernel_size=(2, 2), stride=(2, 2))
        self.conv = DoubleConv(320, 320, drop, conv=conv_l)

    def forward(self, x: torch.Tensor, conc: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x)
        x = torch.cat([conc, x1], dim=1)
        return self.conv(x)

class Up_Block_3p3d(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop: float = 0.1,
        conv_l: nn.Module = nn.Conv3d,
    ):
        """
        Initialize the Up_Block

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param drop: dropout rate
        :type drop: float
        """
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv = DoubleConv3d(320, 320, drop, conv=conv_l)

    def forward(self, x: torch.Tensor, conc: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x)
        x = torch.cat([conc, x1], dim=1)
        return self.conv(x)

class Up_Block_deepmeta(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
    ):
        """
        Initialize the Up_Block

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param drop: dropout rate
        :type drop: float
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, 32, kernel_size=(2, 2), stride=(2, 2))
        self.conv = DoubleConv(160, 160, drop, conv=conv_l)

    def forward(self, x: torch.Tensor, conc: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x)
        x = torch.cat([conc, x1], dim=1)
        return self.conv(x)

class Up_Block_deepmeta3d(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop: float = 0.1,
        conv_l: nn.Module = nn.Conv3d,
    ):
        """
        Initialize the Up_Block

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param drop: dropout rate
        :type drop: float
        """
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv = DoubleConv3d(160, 160, drop, conv=conv_l)

    def forward(self, x: torch.Tensor, conc: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x)
        x = torch.cat([conc, x1], dim=1)
        return self.conv(x)
