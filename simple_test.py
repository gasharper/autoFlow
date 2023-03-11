"""
    A simple python script can quickly learn how to use `autoFlow` framework.
"""

import torch # torch >= 1.9.0
import torch.nn as nn
import torch.nn.functional as F
from autoFlow import InvertibleModule
from autoFlow import SequentialNF


""" Estimate affine layer parameters ,`exp(s)` and `t()` """
class AffineParamBlock(nn.Module):
    def __init__(self, in_ch):
        super(AffineParamBlock, self).__init__()
        self.clamp = 3
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 2*in_ch, kernel_size=7, padding=7//2, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(2*in_ch, 2*in_ch, kernel_size=7, padding=7//2, bias=False),
        )

    def forward(self, input):
        output = self.conv(input)
        _dlogdet, bias = output.chunk(2, 1)
        dlogdet = self.clamp * 0.636 * torch.atan(_dlogdet / self.clamp)  # Soft clipping
        scale = torch.exp(dlogdet)
        return (scale, bias), dlogdet # scale * x + bias

    
""" Single affine coupling layer, there are two kinds of fusion: `up` or `down` """
class FlowBlock(InvertibleModule):
    def __init__(self, channel, direct):
        super(FlowBlock, self).__init__()
        assert direct in ['up', 'down']
        self.direct = direct
        self.affineParams = AffineParamBlock(channel)

    def forward(self, inputs, logdets):
        x0, x1 = inputs
        logdet0, logdet1 = logdets
        if self.direct == 'up':
            y10 = F.interpolate(x1, size=x0.shape[2:], mode='nearest') # interpolation first in up-sampling
            (scale0, bias0), dlogdet0 = self.affineParams(y10)
            z0, z1 = scale0*x0+bias0, x1
            dlogdet1 = 0
        else:
            (scale10, bias10), dlogdet10 = self.affineParams(x0)
            scale1, bias1, dlogdet1 = F.interpolate(scale10, size=x1.shape[2:], mode='nearest'),\
                                         F.interpolate(bias10, size=x1.shape[2:], mode='nearest'),\
                                             F.interpolate(dlogdet10, size=x1.shape[2:], mode='nearest') # interpolation after in down-sampling
            z0, z1 = x0, scale1*x1+bias1
            dlogdet0 = 0
        outputs = (z0, z1)
        out_logdets = (logdet0+dlogdet0, logdet1+dlogdet1)
        return outputs, out_logdets

    def inverse(self, outputs, logdets):
        z0, z1 = outputs
        logdet0, logdet1 = logdets
        if self.direct == 'up':
            z10 = F.interpolate(z1, size=z0.shape[2:], mode='nearest') # interpolation first in up-sampling
            (scale0, bias0), dlogdet0 = self.affineParams(z10)
            x0, x1 = (z0-bias0)/scale0, z1
            dlogdet1 = 0
        else:
            (scale01, bias01), dlogdet01 = self.affineParams(z0)
            scale1, bias1, dlogdet1 = F.interpolate(scale01, size=z1.shape[2:], mode='nearest'),\
                                         F.interpolate(bias01, size=z1.shape[2:], mode='nearest'),\
                                             F.interpolate(dlogdet01, size=z1.shape[2:], mode='nearest') # interpolation after in down-sampling
            x0, x1 = z0, (z1-bias1)/scale1
            dlogdet0 = 0
        inputs = (x0, x1)
        in_logdets = (logdet0-dlogdet0, logdet1-dlogdet1)
        return inputs, in_logdets


""" semi-invertible 1x1Conv """
class Invertible_1x1Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels ) -> None:
        assert out_channels >= in_channels
        super().__init__(in_channels, out_channels, kernel_size=1, bias=False)
    def inverse(self, output):
        b, c, h, w = output.shape
        A = self.weight[..., 0, 0] # outch, inch
        B = output.permute([1,0,2,3]).reshape(c, -1) # outch, bhw
        X = torch.linalg.lstsq(A, B)  # AX=B
        return X.solution.reshape(-1, b, h, w).permute([1, 0, 2, 3])
    @property
    def logdet(self):
        w = self.weight.squeeze() # out,in
        return 0.5*torch.logdet(w.T@w)


""" `filter2d` function from kornia """
def kornia_filter2d(
    input: torch.Tensor, kernel: torch.Tensor, 
) -> torch.Tensor:
    
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)
    height, width = tmp_kernel.shape[-2:]

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    return output


""" Invertible Pyramid (aka LaplacianPyramid) """
class LaplacianPyramid(nn.Module):
    def __init__(self, num_levels) -> None:
        super().__init__()
        self.kernel = torch.tensor(
            [
                [
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [6.0, 24.0, 36.0, 24.0, 6.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                ]
            ]
        )/ 256.0
        self.num_levels = num_levels - 1 # total num_levels layers
                                         #  which the last layer is the last Gaussian pyramid layer.

    def _pyramid_down(self, input, pad_mode='reflect'):
        if not len(input.shape) == 4:
            raise ValueError(f'Invalid img shape, we expect BCHW, got: {input.shape}')
        # blur 
        img_pad = F.pad(input, (2,2,2,2), mode=pad_mode)
        img_blur = kornia_filter2d(img_pad, kernel=self.kernel)
        # downsample
        out = F.interpolate(img_blur, scale_factor=0.5, mode='nearest')
        return out

    def _pyramid_up(self, input, size, pad_mode='reflect'):
        if not len(input.shape) == 4:
            raise ValueError(f'Invalid img shape, we expect BCHW, got: {input.shape}')
        # upsample
        img_up = F.interpolate(input, size=size, mode='nearest', )
        # blur
        img_pad = F.pad(img_up, (2,2,2,2), mode=pad_mode)
        img_blur = kornia_filter2d(img_pad, kernel=self.kernel)
        return img_blur
        
    def build_pyramid(self, input):
        gp, lp = [input], []
        for _ in range(self.num_levels):
            gp.append(self._pyramid_down(gp[-1]))
        for layer in range(self.num_levels):
            curr_gp = gp[layer]
            next_gp = self._pyramid_up(gp[layer+1], size=curr_gp.shape[2:])
            lp.append(curr_gp - next_gp)
        lp.append(gp[self.num_levels])
        return lp

    def compose_pyramid(self, lp):
        rs = lp[-1]
        for i in range(self.num_levels-1, -1, -1):
            rs = self._pyramid_up(rs, size=lp[i].shape[2:])
            rs = torch.add(rs, lp[i])
        return rs


""" The simplest PyramidFlow (w/o Volume Normalization or other tricks) to test `autoFlow` framework. """
class SimpleTestFlow(SequentialNF):
    def __init__(self, modules, channel):
        super().__init__(modules)
        self.inconv = Invertible_1x1Conv(3, channel)
        self.pyramid = LaplacianPyramid(2)

    def forward(self, img):
        feat = self.inconv(img)
        pyramid = self.pyramid.build_pyramid(feat)
        logdets = tuple(torch.zeros_like(pyramid_j) for pyramid_j in pyramid)
        return super().forward(pyramid, logdets)

    def inverse(self, pyramid_out, logdets_out):
        pyramid_in, logdets_in = super().inverse(pyramid_out, logdets_out)
        feat = self.pyramid.compose_pyramid(pyramid_in)
        rev_img = self.inconv.inverse(feat)
        return rev_img, logdets_in


if __name__ == '__main__':
    torch.random.manual_seed(0)
    num_stack = 3
    channel = 64
    
    # Module sequence
    modules = []
    for _ in range(num_stack):
        modules.append(FlowBlock(channel, direct='up'))
        modules.append(FlowBlock(channel, direct='down'))

    # NF
    flowNF = SimpleTestFlow(modules, channel=channel).cuda()
    img = torch.randn(2, 3, 256, 256).cuda()
    pyramid_out, logdets_out = flowNF.forward(img)
    rev_img, _ = flowNF.inverse(pyramid_out, logdets_out)

    # Print results
    dimg = torch.abs(img - rev_img)
    print(f'Maximum Error: {dimg.max().item():.3e}') # 2.146e-06


