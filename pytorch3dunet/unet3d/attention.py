
import torch
from torch import nn
from torch.nn import functional as F
NormLayerGlobal=nn.BatchNorm3d

class SpatialSELayer3d(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer3d, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.conv1 = nn.Conv3d(3,1,kernel_size=7,padding=3,stride=1)
        # self.conv2 = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.norm = NormLayerGlobal(num_channels, affine=True)

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        xconv1 = self.conv(input_tensor)
        x= input_tensor.view(batch_size,channel,D*H*W).permute(0,2,1)
        xavg = self.avg_pool(x)
        xavg = xavg.permute(0,2,1).view(batch_size,1,D,H,W)
        xmax = self.max_pool(x)
        xmax = xmax.permute(0,2,1).view(batch_size,1,D,H,W)
        xconv1 = self.conv1(torch.cat([xavg,xmax,xconv1],1))
            # out2 = self.conv2(input_tensor)
        squeeze_tensor = self.sigmoid(xconv1)
        # squeeze_tensor2 =  torch.tanh(out2)

        input_tensor = self.norm(input_tensor)
        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W)) #+ squeeze_tensor2.view(batch_size,1,D,H,W)

        return output_tensor


class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1,inplace=True)
        self.sigmoid = nn.Sigmoid()
        # self.fc3 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        # self.fc4 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.norm = NormLayerGlobal(num_channels, affine=True)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)
        squeeze_tensor_max = self.max_pool(input_tensor)
        # channel excitation
        fc_out_1 = self.lrelu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.fc2(fc_out_1)
        fc_out_3 = self.lrelu(self.fc1(squeeze_tensor_max.view(batch_size, num_channels)))
        fc_out_4 =self.fc2(fc_out_3)
        fc_out = self.sigmoid(fc_out_2+fc_out_4)
        input_tensor = self.norm(input_tensor)
        output_tensor = torch.mul(input_tensor, fc_out.view(batch_size, num_channels, 1, 1, 1)) #+ fc_out_4.view(batch_size, num_channels, 1, 1, 1)

        return output_tensor


class ChannelSpatialSELayer3d(nn.Module):
    def __init__(self,in_channels):
        super(ChannelSpatialSELayer3d,self).__init__()
        self.sSE = SpatialSELayer3d(in_channels)        
        self.cSE = ChannelSELayer3D(in_channels)

    def forward(self,x):
        # output_tensor = self.sSE(self.cSE(x))
        # output_tensor = self.cSE(self.sSE(x))
        output_tensor = torch.max(self.cSE(x), self.sSE(x))
        return output_tensor
