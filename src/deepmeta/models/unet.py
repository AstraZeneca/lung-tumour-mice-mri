"""unet-like models, implemented and modified from https://github.com/cbib/DeepMeta/tree/master/src/models"""
import torch.nn as nn
import src.deepmeta.models.unet_parts as up

class Unet(nn.Module):
    """
    Implementation of the U-Net model from the paper:
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (Ronneberger et al., 2015)
    """

    def __init__(
        self,
        filters: int = 64,
        classes: int = 1,
        drop_r: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
        dim:str ='2d'
    ):
        """
        Initialize the U-Net model.
        :param filters: Number of filters in the first convolutional layer.
        :type filters: int
        :param classes: Number of classes to predict.
        :type classes: int
        :param drop_r: Dropout rate.
        :type drop_r: float
        """
        super().__init__()
        if dim=='2d':
            Down_Block=up.Down_Block
            Up_Block=up.Up_Block
            Bridge=up.Bridge
            OutConv=up.OutConv
        elif dim=='3d':
            Down_Block=up.Down_Block3d
            Up_Block=up.Up_Block3d
            Bridge=up.Bridge3d
            conv_l=nn.Conv3d
            OutConv=up.OutConv3d
        else:
            raise ValueError('no such value; choose 2d or 3d')
        self.down1 = Down_Block(1, filters, conv_l=conv_l)
        self.down2 = Down_Block(filters, filters * 2, drop_r, conv_l=conv_l)
        self.down3 = Down_Block(filters * 2, filters * 4, drop_r, conv_l=conv_l)
        self.down4 = Down_Block(filters * 4, filters * 8, drop_r, conv_l=conv_l)

        self.bridge = Bridge(filters * 8, filters * 16, drop_r, conv_l=conv_l)

        self.up1 = Up_Block(filters * 16, filters * 8, drop_r, conv_l=conv_l)
        self.up2 = Up_Block(filters * 8, filters * 4, drop_r, conv_l=conv_l)
        self.up3 = Up_Block(filters * 4, filters * 2, drop_r, conv_l=conv_l)
        self.up4 = Up_Block(filters * 2, filters, drop_r, conv_l=conv_l)

        self.outc = up.OutConv(filters, classes, conv_l=conv_l)

    def forward(self, x):
        c1, x1 = self.down1(x)
        c2, x2 = self.down2(x1)
        c3, x3 = self.down3(x2)
        c4, x4 = self.down4(x3)
        bridge = self.bridge(x4)
        x = self.up1(bridge, c4)
        x = self.up2(x, c3)
        x = self.up3(x, c2)
        x = self.up4(x, c1)
        return self.outc(x)


class Unet3plus(
    nn.Module
):
    """
    Implementation of the U-Net model from the paper:
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (Ronneberger et al., 2015)
    """

    def __init__(
        self,
        filters: int = 64,
        classes: int = 2,
        drop_r: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
        dim:str='2d'
    ):
        """
        Initialize the U-Net model.
        :param filters: Number of filters in the first convolutional layer.
        :type filters: int
        :param classes: Number of classes to predict.
        :type classes: int
        :param drop_r: Dropout rate.
        :type drop_r: float
        """
        super().__init__()

        if dim=='2d':
            Down_Block=up.Down_Block
            Up_Block_3p=up.Up_Block_3p
            Concat_Block_3p=up.Concat_Block_3p
            Bridge=up.Bridge
            OutConv=up.OutConv
        elif dim=='3d':
            filters = 32
            Down_Block=up.Down_Block3d
            Up_Block_3p=up.Up_Block_3p3d
            Concat_Block_3p=up.Concat_Block_3p3d
            Bridge=up.Bridge3d
            conv_l=nn.Conv3d
            OutConv=up.OutConv3d
        else:
            raise ValueError('no such value; choose 2d or 3d')

        self.down1 = Down_Block(1, filters, conv_l=conv_l)
        self.down2 = Down_Block(filters, filters * 2, drop_r, conv_l=conv_l)
        self.down3 = Down_Block(filters * 2, filters * 4, drop_r, conv_l=conv_l)
        self.down4 = Down_Block(filters * 4, filters * 8, drop_r, conv_l=conv_l)

        self.bridge = Bridge(filters * 8, filters * 16, drop_r, conv_l=conv_l)

        self.up1 = Up_Block_3p(filters * 16, 320, drop_r, conv_l=conv_l)
        self.up2 = Up_Block_3p(320, 320, drop_r, conv_l=conv_l)
        self.up3 = Up_Block_3p(320, 320, drop_r, conv_l=conv_l)
        self.up4 = Up_Block_3p(320, 320, drop_r, conv_l=conv_l)

        self.concat1 = Concat_Block_3p(
            kernels_down=[8, 4, 2, 1],
            filters_down=[filters, filters * 2, filters * 4, filters * 8],
            kernels_up=[],
            filters_up=[],
        )
        self.concat2 = Concat_Block_3p(
            kernels_down=[4, 2, 1],
            filters_down=[filters, filters * 2, filters * 4],
            kernels_up=[4],
            filters_up=[filters * 16],
        )
        self.concat3 = Concat_Block_3p(
            kernels_down=[2, 1],
            filters_down=[filters, filters * 2],
            kernels_up=[4, 8],
            filters_up=[320, filters * 16],
        )
        self.concat4 = Concat_Block_3p(
            kernels_down=[1],
            filters_down=[filters],
            kernels_up=[4, 8, 16],
            filters_up=[320, 320, filters * 16],
        )
        self.outc = OutConv(320, classes, conv_l=conv_l)

    def forward(self, x):
        c1, x1 = self.down1(x)
        c2, x2 = self.down2(x1)
        c3, x3 = self.down3(x2)
        c4, x4 = self.down4(x3)
        bridge = self.bridge(x4)
        x5 = self.up1(bridge, self.concat1([c1, c2, c3, c4], []))
        x6 = self.up2(x5, self.concat2([c1, c2, c3], [bridge]))
        x7 = self.up3(x6, self.concat3([c1, c2], [x5, bridge]))
        x8 = self.up4(x7, self.concat4([c1], [x6, x5, bridge]))
        return self.outc(x8)

class DeepMeta(
    nn.Module
):
    """
    Implementation of the U-Net model from the paper:
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (Ronneberger et al., 2015)
    """

    def __init__(
        self,
        filters: int = 16,
        classes: int = 2,
        drop_r: float = 0.1,
        conv_l: nn.Module = up.SeparableConv2d,
        dim:str='2d'
    ):
        """
        Initialize the U-Net model.
        :param filters: Number of filters in the first convolutional layer.
        :type filters: int
        :param classes: Number of classes to predict.
        :type classes: int
        :param drop_r: Dropout rate.
        :type drop_r: float
        """
        super().__init__()
        if dim=='2d':
            Down_Block=up.Down_Block
            Up_Block_3p=up.Up_Block_deepmeta
            Concat_Block=up.Concat_Block
            Bridge=up.Bridge
            OutConv=up.OutConv
        elif dim=='3d':
            Down_Block=up.Down_Block3d
            Up_Block_3p=up.Up_Block_deepmeta3d
            Concat_Block=up.Concat_Block3d
            Bridge=up.Bridge3d
            conv_l=up.SeparableConv3d
            OutConv=up.OutConv3d
        else:
            raise ValueError('no such value; choose 2d or 3d')

        self.down1 = Down_Block(1, filters, conv_l=conv_l)
        self.down2 = Down_Block(filters, filters * 2, drop_r, conv_l=conv_l)
        self.down3 = Down_Block(filters * 2, filters * 4, drop_r, conv_l=conv_l)
        self.down4 = Down_Block(filters * 4, filters * 8, drop_r, conv_l=conv_l)

        self.bridge = Bridge(filters * 8, filters * 16, drop_r, conv_l=conv_l)

        self.up1 = Up_Block_3p(filters * 16, 160, drop_r, conv_l=conv_l)
        self.up2 = Up_Block_3p(160, 160, drop_r, conv_l=conv_l)
        self.up3 = Up_Block_3p(160, 160, drop_r, conv_l=conv_l)
        self.up4 = Up_Block_3p(160, 160, drop_r, conv_l=conv_l)

        self.concat1 = Concat_Block(
            kernels_down=[8, 4, 2, 1],
            filters_down=[filters, filters * 2, filters * 4, filters * 8],
            kernels_up=[],
            filters_up=[],
        )
        self.concat2 = Concat_Block(
            kernels_down=[4, 2, 1],
            filters_down=[filters, filters * 2, filters * 4],
            kernels_up=[4],
            filters_up=[filters * 16],
        )
        self.concat3 = Concat_Block(
            kernels_down=[2, 1],
            filters_down=[filters, filters * 2],
            kernels_up=[4, 8],
            filters_up=[160, filters * 16],
        )
        self.concat4 = Concat_Block(
            kernels_down=[1],
            filters_down=[filters],
            kernels_up=[4, 8, 16],
            filters_up=[160, 160, filters * 16],
        )
        self.outc = OutConv(160, classes, conv_l=conv_l)

    def forward(self, x):
        c1, x1 = self.down1(x)
        c2, x2 = self.down2(x1)
        c3, x3 = self.down3(x2)
        c4, x4 = self.down4(x3)
        bridge = self.bridge(x4)
        x5 = self.up1(bridge, self.concat1([c1, c2, c3, c4], []))
        x6 = self.up2(x5, self.concat2([c1, c2, c3], [bridge]))
        x7 = self.up3(x6, self.concat3([c1, c2], [x5, bridge]))
        x8 = self.up4(x7, self.concat4([c1], [x6, x5, bridge]))
        return self.outc(x8)
