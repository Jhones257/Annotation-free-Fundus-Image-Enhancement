# -*- coding: UTF-8 -*-
"""
@Function:
@File: gfenet_backbone.py
@Date: 2023/6/27 14:49 
@Author: Hever
"""
import torch
import torch.nn as nn
import functools


def _use_bias_from_norm(norm_layer):
    if type(norm_layer) == functools.partial:
        return norm_layer.func == nn.InstanceNorm2d
    return norm_layer == nn.InstanceNorm2d


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, norm_layer=nn.BatchNorm2d):
        super(ResidualConvBlock, self).__init__()
        use_bias = _use_bias_from_norm(norm_layer)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.norm1 = norm_layer(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.norm2 = norm_layer(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResidualDownBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d):
        super(ResidualDownBlock, self).__init__()
        use_bias = _use_bias_from_norm(norm_layer)
        self.conv1 = nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.norm1 = norm_layer(output_nc)
        self.conv2 = nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.norm2 = norm_layer(output_nc)
        self.skip = nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=2, padding=0, bias=use_bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class AttentionGate(nn.Module):
    def __init__(self, g_channels, x_channels, inter_channels, norm_layer=nn.BatchNorm2d):
        super(AttentionGate, self).__init__()
        use_bias = _use_bias_from_norm(norm_layer)
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        attention = self.psi(self.W_g(g) + self.W_x(x))
        return x * attention


class UnetGFENetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, use_residual_blocks=False, use_attention=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGFENetGenerator, self).__init__()
        assert num_downs == 8
        unet_block8 = GFENetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, norm_layer=norm_layer,
                            innermost=True, use_residual_blocks=use_residual_blocks)  # add the innermost layer
        unet_block7 = GFENetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                             norm_layer=norm_layer, use_dropout=use_dropout,
                             use_residual_blocks=use_residual_blocks)
        unet_block6 = GFENetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                             norm_layer=norm_layer, use_dropout=use_dropout,
                             use_residual_blocks=use_residual_blocks)
        unet_block5 = GFENetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                             norm_layer=norm_layer, use_dropout=use_dropout,
                             use_residual_blocks=use_residual_blocks)
        unet_block4 = GFENetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, norm_layer=norm_layer,
                            use_residual_blocks=use_residual_blocks)
        unet_block3 = GFENetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, norm_layer=norm_layer,
                            use_residual_blocks=use_residual_blocks)
        unet_block2 = GFENetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, norm_layer=norm_layer,
                            use_residual_blocks=use_residual_blocks)
        unet_block1 = GFENetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, outermost=True,
                            norm_layer=norm_layer,
                            use_residual_blocks=use_residual_blocks)  # add the outermost layer

        self.down1, self.up1, self.h_up1 = unet_block1.down, unet_block1.up, unet_block1.h_up
        self.down2, self.up2, self.h_up2 = unet_block2.down, unet_block2.up, unet_block2.h_up
        self.down3, self.up3, self.h_up3 = unet_block3.down, unet_block3.up, unet_block3.h_up
        self.down4, self.up4, self.h_up4 = unet_block4.down, unet_block4.up, unet_block4.h_up
        self.down5, self.up5, self.h_up5 = unet_block5.down, unet_block5.up, unet_block5.h_up
        self.down6, self.up6, self.h_up6 = unet_block6.down, unet_block6.up, unet_block6.h_up
        self.down7, self.up7, self.h_up7 = unet_block7.down, unet_block7.up, unet_block7.h_up
        self.down8, self.up8, self.h_up8 = unet_block8.down, unet_block8.up, unet_block8.h_up

        self.use_attention = use_attention
        if self.use_attention:
            self.attn7 = AttentionGate(ngf * 8, ngf * 8, ngf * 4, norm_layer=norm_layer)
            self.attn6 = AttentionGate(ngf * 8, ngf * 8, ngf * 4, norm_layer=norm_layer)
            self.attn5 = AttentionGate(ngf * 8, ngf * 8, ngf * 4, norm_layer=norm_layer)
            self.attn4 = AttentionGate(ngf * 8, ngf * 8, ngf * 4, norm_layer=norm_layer)
            self.attn3 = AttentionGate(ngf * 4, ngf * 4, ngf * 2, norm_layer=norm_layer)
            self.attn2 = AttentionGate(ngf * 2, ngf * 2, ngf, norm_layer=norm_layer)
            self.attn1 = AttentionGate(ngf, ngf, max(1, ngf // 2), norm_layer=norm_layer)


    def forward(self, x):
        """Standard forward"""
        # downsample
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # upsample
        h_u8 = self.up8(d8)
        skip7 = self.attn7(h_u8, d7) if self.use_attention else d7
        h_u7 = self.up7(torch.cat([h_u8, skip7], 1))
        skip6 = self.attn6(h_u7, d6) if self.use_attention else d6
        h_u6 = self.up6(torch.cat([h_u7, skip6], 1))
        skip5 = self.attn5(h_u6, d5) if self.use_attention else d5
        h_u5 = self.up5(torch.cat([h_u6, skip5], 1))
        skip4 = self.attn4(h_u5, d4) if self.use_attention else d4
        h_u4 = self.up4(torch.cat([h_u5, skip4], 1))
        skip3 = self.attn3(h_u4, d3) if self.use_attention else d3
        h_u3 = self.up3(torch.cat([h_u4, skip3], 1))
        skip2 = self.attn2(h_u3, d2) if self.use_attention else d2
        h_u2 = self.up2(torch.cat([h_u3, skip2], 1))
        skip1 = self.attn1(h_u2, d1) if self.use_attention else d1
        h_u1 = self.up1(torch.cat([h_u2, skip1], 1))

        # # upsample
        u8 = self.h_up8(d8)
        u7 = self.h_up7(torch.cat([h_u8, u8], 1))
        u6 = self.h_up6(torch.cat([h_u7, u7], 1))
        u5 = self.h_up5(torch.cat([h_u6, u6], 1))
        u4 = self.h_up4(torch.cat([h_u5, u5], 1))
        u3 = self.h_up3(torch.cat([h_u4, u4], 1))
        u2 = self.h_up2(torch.cat([h_u3, u3], 1))
        u1 = self.h_up1(torch.cat([h_u2, u2], 1))
        # return u1
        return h_u1, u1



class GFENetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, use_residual_blocks=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(GFENetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = _use_bias_from_norm(norm_layer)
        if input_nc is None:
            input_nc = outer_nc

        if use_residual_blocks:
            downconv = ResidualDownBlock(input_nc, inner_nc, norm_layer=norm_layer)
        else:
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                 stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)
        h_uprelu = nn.ReLU()
        h_upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, input_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            # 注意：仅仅修改了这个
            h_upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv] if use_residual_blocks else [downconv]
            if use_residual_blocks:
                up = [uprelu, upconv, ResidualConvBlock(input_nc, norm_layer=norm_layer), nn.Tanh()]
                h_up = [h_uprelu, h_upconv, ResidualConvBlock(outer_nc, norm_layer=norm_layer), nn.Tanh()]
            else:
                up = [uprelu, upconv, nn.Tanh()]
                h_up = [h_uprelu, h_upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            h_upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            if use_residual_blocks:
                down = [downconv]
                up = [uprelu, upconv, upnorm, ResidualConvBlock(outer_nc, norm_layer=norm_layer)]
                h_up = [h_uprelu, h_upconv, h_upnorm, ResidualConvBlock(outer_nc, norm_layer=norm_layer)]
            else:
                down = [downrelu, downconv]
                up = [uprelu, upconv, upnorm]
                h_up = [h_uprelu, h_upconv, h_upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            h_upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            if use_residual_blocks:
                down = [downconv]
                up = [uprelu, upconv, upnorm, ResidualConvBlock(outer_nc, norm_layer=norm_layer)]
                h_up = [h_uprelu, h_upconv, h_upnorm, ResidualConvBlock(outer_nc, norm_layer=norm_layer)]
            else:
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upconv, upnorm]
                h_up = [h_uprelu, h_upconv, h_upnorm]
            if use_dropout:
                up = up + [nn.Dropout(0.5)]
                h_up = h_up + [nn.Dropout(0.5)]
        self.up = nn.Sequential(*up)
        self.h_up = nn.Sequential(*h_up)
        self.down = nn.Sequential(*down)