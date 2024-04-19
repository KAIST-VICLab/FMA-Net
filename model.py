import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class DynamicDownsampling(torch.nn.Module):
    # dynamic filtering for downsampling
    def __init__(self, kernel_size, stride):
        super(DynamicDownsampling, self).__init__()
        # stride = downsampling scale factor
        self.kernel_size = kernel_size
        self.stride = stride

    def kernel_normalize(self, kernel):
        # kernel: [B, H, W, T*k*k]
        return F.softmax(kernel, dim=-1)

    def forward(self, x, kernel, DT):
        # x: [B, C, T, H*stride, W*stride]
        # kernel: [B, k*k, T, H, W]
        # return: [B, C, H, W]

        b, _, t, h, w = kernel.shape
        kernel = kernel.permute(0, 3, 4, 2, 1).contiguous()  # [B, H, W, T, k*k]
        kernel = kernel.view(b, h, w, t * self.kernel_size * self.kernel_size)  # [B, H, W, T*k*k]
        kernel = self.kernel_normalize(kernel)

        kernel = kernel.unsqueeze(dim=1)  # [B, 1, H, W, T*k*k]

        num_pad = (self.kernel_size - self.stride) // 2
        x = F.pad(x, (num_pad, num_pad, num_pad, num_pad, 0, 0), mode="replicate")
        x = x.unfold(3, self.kernel_size, self.stride)
        x = x.unfold(4, self.kernel_size, self.stride)  # [B, C, T, H, W, k, k]
        x = x.permute(0, 1, 3, 4, 2, 5, 6).contiguous()  # [B, C, H, W, T, k, k]
        x = x.view(b, -1, h, w, t * self.kernel_size * self.kernel_size)  # [B, C, H, W, T*k*k]

        x = x * kernel
        x = torch.sum(x, -1)  # [B, C, H, W]

        # normalize
        DT = F.pad(DT, (num_pad, num_pad, num_pad, num_pad, 0, 0), mode="replicate")
        DT = DT.unfold(3, self.kernel_size, self.stride)
        DT = DT.unfold(4, self.kernel_size, self.stride)  # [B, C, T, H, W, k, k]
        DT = DT.permute(0, 1, 3, 4, 2, 5, 6).contiguous()  # [B, C, H, W, T, k, k]
        DT = DT.view(b, -1, h, w, t * self.kernel_size * self.kernel_size)  # [B, C, H, W, T*k*k]

        DT = DT * kernel
        DT = torch.sum(DT, -1)  # [B, C, H, W]

        x = x / (DT + 1e-8)

        return x


class DynamicUpampling(torch.nn.Module):
    # dynamic filtering for upsampling
    def __init__(self, kernel_size, scale):
        super(DynamicUpampling, self).__init__()
        # stride = downsampling scale factor
        self.kernel_size = kernel_size
        self.scale = scale

    def kernel_normalize(self, kernel):
        # kernel: [B, C, H, W, T*k*k]
        K = kernel.shape[-1]
        kernel = kernel - torch.mean(kernel, dim=-1, keepdim=True)
        kernel = kernel + 1.0 / K

        return kernel

    def forward(self, x, kernel):
        # x: [B, C, T, H, W]
        # kernel: [B, s*s*k*k, T, H, W]
        # return: [B, C, H*s, W*s]

        b, c, t, h, w = x.shape

        kernel = rearrange(kernel, 'b (s1 s2 k1 k2) t h w -> b h w s1 s2 (t k1 k2)', s1=self.scale, s2=self.scale, k1=self.kernel_size, k2=self.kernel_size)
        kernel = kernel.unsqueeze(dim=1)
        kernel = self.kernel_normalize(kernel)

        num_pad = self.kernel_size // 2
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = F.pad(x, (num_pad, num_pad, num_pad, num_pad), mode="replicate")
        x = F.unfold(x, [self.kernel_size, self.kernel_size], padding=0)
        x = rearrange(x, '(b t) (c k1 k2) (h w) -> b c h w (t k1 k2)', b=b, t=t, c=c, k1=self.kernel_size, k2=self.kernel_size, h=h, w=w)
        x = x.unsqueeze(dim=4).unsqueeze(dim=5)
        x = torch.sum(x * kernel, dim=-1) # [B, C, H, W, s, s]
        x = x.permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(b, c, self.scale*h, self.scale*w)

        return x


def backwarp(x, flow, objBackwarpcache):
    # x: [B, C, H, W]
    # flow: [B, 2, H, W]
    if 'grid' + str(flow.dtype) + str(flow.device) + str(flow.shape[2]) + str(flow.shape[3]) not in objBackwarpcache:
        tenHor = torch.linspace(start=-1.0, end=1.0, steps=flow.shape[3], dtype=flow.dtype,
                                device=flow.device).view(1, 1, 1, -1).repeat(1, 1, flow.shape[2], 1)
        tenVer = torch.linspace(start=-1.0, end=1.0, steps=flow.shape[2], dtype=flow.dtype,
                                device=flow.device).view(1, 1, -1, 1).repeat(1, 1, 1, flow.shape[3])

        objBackwarpcache['grid' + str(flow.dtype) + str(flow.device) + str(flow.shape[2]) + str(flow.shape[3])] = torch.cat([tenHor, tenVer], 1)

    if flow.shape[3] == flow.shape[2]:
        flow = flow * (2.0 / ((flow.shape[3] and flow.shape[2]) - 1.0))

    elif flow.shape[3] != flow.shape[2]:
        flow = flow * torch.tensor(data=[2.0 / (flow.shape[3] - 1.0), 2.0 / (flow.shape[2] - 1.0)], dtype=flow.dtype, device=flow.device).view(1, 2, 1, 1)

    return nn.functional.grid_sample(input=x, grid=(objBackwarpcache['grid' + str(flow.dtype) + str(flow.device) + str(flow.shape[2]) + str(flow.shape[3])] + flow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)


class ImageBWarp(torch.nn.Module):
    def __init__(self, scale, num_seq):
        super(ImageBWarp, self).__init__()
        self.scale = scale
        self.num_seq = num_seq
        self.objBackwarpcache = {}
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, f):
        # x: [B, 3, T, H, W]
        # f: [B, 2+1, T, H, W]

        x = rearrange(x, 'b c t h w -> (b t) c h w')  # [B*T, 3, sH, sW]
        f = rearrange(f, 'b c t h w -> (b t) c h w')  # [B*T, 2+1, H, W]

        weight = f[:, 2:3, :, :]  # [b, 1, h, w]
        flow = f[:, :2, :, :]  # [b, 2, h, w]

        weight = self.sigmoid(weight)

        ones = torch.ones_like(x)

        if self.scale != 1:
            flow = self.scale * F.interpolate(flow, scale_factor=(self.scale, self.scale), mode='bilinear', align_corners=False)
            weight = F.interpolate(weight, scale_factor=(self.scale, self.scale), mode='bilinear', align_corners=False)

        x = backwarp(x, flow, self.objBackwarpcache)
        ones = backwarp(ones, flow, self.objBackwarpcache)
        ones = ones * weight

        x = rearrange(x, '(b t) c h w -> b c t h w', t=self.num_seq)  # [B, C, T, H, W]
        ones = rearrange(ones, '(b t) c h w -> b c t h w', t=self.num_seq)  # [B, C, T, H, W]
        # flow = rearrange(flow, '(b t) c h w -> b c t h w', t=self.num_seq)  # [B, C, T, H, W]
        weight = rearrange(weight, '(b t) c h w -> b c t h w', t=self.num_seq)  # [B, C, T, H, W]

        return ones, x * weight


class MultiFlowBWarp(torch.nn.Module):
    def __init__(self, dim, num_seq, num_flow):
        super(MultiFlowBWarp, self).__init__()
        self.dim = dim
        self.num_seq = num_seq
        self.num_flow = num_flow
        self.objBackwarpcache = {}
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, F, f):
        # F: [B, C*num_flow, T, H, W]
        # f: [B, 3*num_flow, T, H, W]

        F = rearrange(F, 'b (n c) t h w -> (b n t) c h w', c=self.dim//self.num_flow, n=self.num_flow)    # [B*num_flow*T, C//num_flow, H, W]
        f = rearrange(f, 'b (n c) t h w -> (b n t) c h w', c=3, n=self.num_flow)    # [B*num_flow*T, 2+1, H, W]

        weight = f[:, 2:3, :, :]  # [b, 1, h, w]
        flow = f[:, :2, :, :]  # [b, 2, h, w]

        weight = self.sigmoid(weight)

        F = backwarp(F, flow, self.objBackwarpcache)
        F = F * weight

        F = rearrange(F, '(b n t) c h w -> b (n c) t h w', t=self.num_seq, n=self.num_flow)    # [B, C, T, H, W]

        return F


class PixelShuffleBlock(torch.nn.Module):
    def __init__(self, channels, bias):
        super(PixelShuffleBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1, stride=1, bias=bias)
        self.conv2 = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1, stride=1, bias=bias)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1, bias=bias)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.shuffle = torch.nn.PixelShuffle(2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.shuffle(x)
        x = self.relu(self.conv2(x))
        x = self.shuffle(x)
        x = self.relu(self.conv3(x))

        return x


class DenseLayer(torch.nn.Module):
    def __init__(self, dim, growth_rate, bias):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv3d(dim, growth_rate, kernel_size=[1,3,3], padding=[0,1,1], stride=1, bias=bias)
        self.lrelu = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.lrelu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class RDB(torch.nn.Module):
    def __init__(self, dim, growth_rate, num_dense_layer, bias):
        super(RDB, self).__init__()
        self.layer = [DenseLayer(dim=dim+growth_rate*i, growth_rate=growth_rate, bias=bias) for i in range(num_dense_layer)]
        self.layer = torch.nn.Sequential(*self.layer)
        self.conv = nn.Conv3d(dim+growth_rate*num_dense_layer, dim, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        out = self.layer(x)
        out = self.conv(out)
        out = out + x

        return out


class RRDB(nn.Module):
    def __init__(self, dim, num_RDB, growth_rate, num_dense_layer, bias):
        super(RRDB, self).__init__()
        self.RDBs = nn.ModuleList([RDB(dim=dim, growth_rate=growth_rate, num_dense_layer=num_dense_layer, bias=bias) for _ in range(num_RDB)])
        self.conv = nn.Sequential(*[nn.Conv3d(dim * num_RDB, dim, kernel_size=1, padding=0, stride=1, bias=bias),
                                    nn.Conv3d(dim, dim, kernel_size=[1,3,3], padding=[0,1,1], stride=1, bias=bias)])

    def forward(self, x):
        input = x
        RDBs_out = []
        for rdb_block in self.RDBs:
            x = rdb_block(x)
            RDBs_out.append(x)
        x = self.conv(torch.cat(RDBs_out, dim=1))
        return x + input


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    # Restormer (CVPR 2022) transposed-attnetion block
    # original source code: https://github.com/swz30/Restormer
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_conv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, f):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(f))
        kv = self.kv_dwconv(self.kv_conv(x))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out


class MultiAttentionBlock(torch.nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type, ffn_expansion_factor, bias, is_DA):
        super(MultiAttentionBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.co_attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)

        if is_DA:
            self.norm3 = LayerNorm(dim, LayerNorm_type)
            self.da_attn = Attention(dim, num_heads, bias)
            self.norm4 = LayerNorm(dim, LayerNorm_type)
            self.ffn2 = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, Fw, F0_c, Kd):
        Fw = Fw + self.co_attn(self.norm1(Fw), F0_c)
        Fw = Fw + self.ffn1(self.norm2(Fw))

        if Kd is not None:
            Fw = Fw + self.da_attn(self.norm3(Fw), Kd)
            Fw = Fw + self.ffn2(self.norm4(Fw))

        return Fw


class FRMA(torch.nn.Module):
    def __init__(self, dim, num_seq, growth_rate, num_dense_layer, num_flow, num_multi_attn, num_heads,
                 LayerNorm_type, ffn_expansion_factor, bias, is_DA=False, is_first_f=False, is_first_Fw=False):
        super(FRMA, self).__init__()
        self.rdb = RDB(dim, growth_rate, num_dense_layer, bias)
        self.rdb_KD = RDB(dim, growth_rate, num_dense_layer, bias) if is_DA else None
        self.conv_KD = nn.Conv2d(dim*num_seq, dim, kernel_size=1, padding=0, stride=1, bias=bias) if is_DA else None

        self.bwarp = MultiFlowBWarp(dim, num_seq, num_flow)
        self.conv_Fw = nn.Conv2d(dim*num_seq if is_first_Fw else dim+dim*num_seq, dim, kernel_size=1, padding=0, stride=1, bias=bias)
        self.conv_f = nn.Sequential(nn.Conv3d(dim*2 if is_first_f else dim*2+3*num_flow, 3*num_flow, kernel_size=[1,3,3], padding=[0,1,1], stride=1, bias=bias),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv3d(3*num_flow, 3*num_flow, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias))

        self.multi_attn_block = nn.ModuleList([MultiAttentionBlock(dim, num_heads, LayerNorm_type, ffn_expansion_factor, bias, is_DA) for _ in range(num_multi_attn)])

    def forward(self, F, Fw, f, F0_c, KD=None):
        B, C, T, H, W = F.shape
        F = self.rdb(F)

        if f is not None:
            warped_F = self.bwarp(F, f)
            f = f + self.conv_f(torch.cat([F0_c.repeat([1,1,T,1,1]), f, warped_F], dim=1))
        else:
            f = self.conv_f(torch.cat([F0_c.repeat([1,1,T,1,1]), F], dim=1))

        warped_F = self.bwarp(F, f)
        warped_F = rearrange(warped_F, 'b c t h w -> b (c t) h w')
        if Fw is not None:
            Fw = self.conv_Fw(torch.cat([Fw, warped_F], dim=1))
        else:
            Fw = self.conv_Fw(warped_F)

        if KD is not None:
            KD = self.rdb_KD(KD)
            KD = rearrange(KD, 'b c t h w -> b (c t) h w')
            KD = self.conv_KD(KD)

        for blk in self.multi_attn_block:
            Fw = blk(Fw, F0_c.squeeze(dim=2), KD)

        return F, Fw, f


class Net_D(torch.nn.Module):
    # Degradation Learning Network
    def __init__(self, config):
        super(Net_D, self).__init__()
        self.dim = config.dim
        in_channels = config.in_channels
        dim = config.dim
        num_seq = config.num_seq
        ds_kernel_size = config.ds_kernel_size
        num_RDB = config.num_RDB
        growth_rate = config.growth_rate
        num_dense_layer = config.num_dense_layer
        num_flow = config.num_flow
        num_FRMA = config.num_FRMA
        num_transformer_block = config.num_transformer_block
        num_heads = config.num_heads
        LayerNorm_type = config.LayerNorm_type
        ffn_expansion_factor = config.ffn_expansion_factor
        bias = config.bias

        self.feature_extractor = nn.Sequential(nn.Conv3d(in_channels, dim, kernel_size=[1,3,3], padding=[0,1,1], stride=1, bias=bias),
                                               nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                               RRDB(dim=dim, num_RDB=num_RDB, growth_rate=growth_rate, num_dense_layer=num_dense_layer, bias=config.bias))

        # FRMA blocks for drgradation learning network
        self.FRMA_blocks = nn.ModuleList([FRMA(dim, num_seq, growth_rate, num_dense_layer, num_flow, num_transformer_block, num_heads,
                                               LayerNorm_type, ffn_expansion_factor, bias, is_DA=False, is_first_f=True if i==0 else False,
                                               is_first_Fw=True if i==0 else False) for i in range(num_FRMA)])

        # generate image flow-mask pair for Y
        self.f_conv = nn.Sequential(nn.Conv3d(3*num_flow, 3*num_flow, kernel_size=[1,3,3], padding=[0,1,1], stride=1, bias=bias),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv3d(3*num_flow, 3, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias))

        # generate degradation kernels
        self.d_conv = nn.Sequential(nn.Conv3d(dim//num_seq, dim, kernel_size=[1,3,3], padding=[0,1,1], stride=1, bias=bias),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv3d(dim, ds_kernel_size*ds_kernel_size, kernel_size=[1,3,3], padding=[0,1,1], stride=1, bias=bias))

        # generate anchor for TA loss
        self.a_conv = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                     nn.Conv3d(dim, in_channels, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias))

    def forward(self, x):
        # x: [B, 3, T, H, W]

        B, C, T, H, W = x.shape

        F = self.feature_extractor(x)
        F0_c = F[:, :, T//2:T//2+1, :, :]

        Fw = None
        f = None

        for blk in self.FRMA_blocks:
            F, Fw, f = blk(F, Fw, f, F0_c)

        Fw = rearrange(Fw, 'b (c t) h w -> b c t h w', t=T, c=self.dim // T)
        KD = self.d_conv(Fw)

        f_Y = self.f_conv(f)
        anchor = self.a_conv(F)

        return F, KD, f_Y, f, anchor


class Net_R(torch.nn.Module):
    # Restoration Network
    def __init__(self, config):
        super(Net_R, self).__init__()
        in_channels = config.in_channels
        dim = config.dim
        num_seq = config.num_seq
        ds_kernel_size = config.ds_kernel_size
        us_kernel_size = config.us_kernel_size
        num_RDB = config.num_RDB
        growth_rate = config.growth_rate
        num_dense_layer = config.num_dense_layer
        num_flow = config.num_flow
        num_FRMA = config.num_FRMA
        num_transformer_block = config.num_transformer_block
        num_heads = config.num_heads
        LayerNorm_type = config.LayerNorm_type
        ffn_expansion_factor = config.ffn_expansion_factor
        bias = config.bias
        scale = config.scale

        self.feature_extractor = nn.Sequential(nn.Conv3d(in_channels+dim, dim, kernel_size=[1,3,3], padding=[0,1,1], stride=1, bias=bias),
                                               nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                               RRDB(dim=dim, num_RDB=num_RDB, growth_rate=growth_rate, num_dense_layer=num_dense_layer, bias=config.bias))

        # FRMA blocks for restoration network
        self.FRMA_blocks = nn.ModuleList([FRMA(dim, num_seq, growth_rate, num_dense_layer, num_flow, num_transformer_block, num_heads, LayerNorm_type,
                                               ffn_expansion_factor, bias, is_DA=True, is_first_Fw=True if i==0 else False) for i in range(num_FRMA)])

        self.f_conv1 = nn.Sequential(nn.Conv3d(3*num_flow, 3*num_flow, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv3d(3*num_flow, 3*num_flow, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias))

        self.d_conv = nn.Sequential(nn.Conv3d(ds_kernel_size * ds_kernel_size, dim, kernel_size=3, padding=1, stride=1, bias=bias),
                                              nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                              nn.Conv3d(dim, dim, kernel_size=3, padding=1, stride=1, bias=bias))

        self.res_conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, bias=bias)
        self.res_conv2 = nn.Conv2d(dim, in_channels, kernel_size=3, padding=1, stride=1, bias=bias)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.upsample = PixelShuffleBlock(dim, bias=bias)

        # generate restoration kernels
        self.r_conv = nn.Sequential(nn.Conv3d(dim//num_seq, dim, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv3d(dim, us_kernel_size*us_kernel_size*scale*scale, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias))

        self.f_conv2 = nn.Sequential(nn.Conv3d(3*num_flow, 3*num_flow, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                     nn.Conv3d(3*num_flow, 3, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias))

        self.f_conv3 = nn.Sequential(nn.Conv3d(3*num_flow+3+3, 3*num_flow, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                     nn.Conv3d(3 * num_flow, 3, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias))

        self.bwarp = ImageBWarp(1, num_seq)
        self.duf = DynamicUpampling(us_kernel_size, scale)

        # generate anchor for TA loss
        self.a_conv = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv3d(dim, in_channels, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias))

    def forward(self, x, F, f, KD):
        # x: [B, 3, T, H, W]
        # F: [B, C, T, H, W]
        # f: [B, 3*num_flow, T, H, W]
        # KD: [B, k*k, T, H, W]

        B, C, T, H, W = x.shape

        F = self.feature_extractor(torch.cat([x, F], dim=1))
        F0_c = F[:, :, T//2:T//2+1, :, :]

        Fw = None
        f = self.f_conv1(f)
        KD = self.d_conv(KD)

        for blk in self.FRMA_blocks:
            F, Fw, f = blk(F, Fw, f, F0_c, KD)

        # pixel shuffle upsample
        res = self.relu(self.res_conv1(Fw))
        res = self.upsample(res)
        res = self.res_conv2(res)

        KR = rearrange(Fw, 'b (c t) h w -> b c t h w', t=T)
        KR = self.r_conv(KR)

        f_X = self.f_conv2(f)
        _, warped_X = self.bwarp(x, f_X)
        f_X = self.f_conv3(torch.cat([f, warped_X, x[:, :, T // 2:T // 2 + 1, :, :].repeat([1, 1, T, 1, 1])], dim=1))

        # flow-guided dynamic upsampling
        _, warped_X = self.bwarp(x, f_X)
        output = self.duf(warped_X, KR) + res
        anchor = self.a_conv(F)

        return output, warped_X, anchor


class FMANet(torch.nn.Module):
    def __init__(self, config):
        super(FMANet, self).__init__()

        self.stage = config.stage
        self.degradation_learning_network = Net_D(config)
        self.bwarp = ImageBWarp(config.scale, config.num_seq)
        self.ddf = DynamicDownsampling(config.ds_kernel_size, config.scale)

        if self.stage == 2:
            self.restoration_network = Net_R(config)

    def forward(self, x, y=None):
        # x: [B, 3, T, H, W]
        # y: [B, 3, T, sH, sW]

        result_dict = {}

        F, KD, f_Y, f, anchor_D = self.degradation_learning_network(x)

        if y is not None:
            # flow-guided dynamic downsampling
            ones, warped_Y = self.bwarp(y, f_Y)
            recon = self.ddf(warped_Y, KD, ones)

            result_dict['recon'] = recon
            result_dict['hr_warp'] = warped_Y
            result_dict['image_flow'] = f_Y[:, :2, :, :, :]
            result_dict['F_sharp_D'] = anchor_D

        if self.stage == 1:
            return result_dict

        elif self.stage == 2:
            output, warped_X, anchor_R = self.restoration_network(x, F, f, KD)
            result_dict['output'] = output
            result_dict['lr_warp'] = warped_X
            result_dict['F_sharp_R'] = anchor_R

            return result_dict