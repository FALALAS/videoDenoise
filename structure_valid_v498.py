import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_pyramid import _get_pyramid_gaussian_kernel

C_all = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def space_to_depth(in_tensor, down_scale):
    n, c, h, w = in_tensor.size()
    unfolded_x = F.unfold(in_tensor, down_scale, stride=down_scale)
    return unfolded_x.view(n, c * down_scale ** 2, h // down_scale, w // down_scale)


class pyrdown(torch.nn.Module):
    def __init__(self, channels):
        super(pyrdown, self).__init__()

        self.channels = channels

        self.net = nn.Conv2d(channels, channels, kernel_size=5, groups=channels, padding=2, stride=1, bias=False)
        kernel = _get_pyramid_gaussian_kernel().unsqueeze(1).expand(channels, -1, -1, -1).clone()
        self.net.weight = torch.nn.Parameter(kernel)

    def forward(self, x):
        b, c, h, w = x.shape
        x_blur = self.net(x)
        out = F.interpolate(x_blur, size=(int(float(h) / 2), int(float(w) // 2)), mode='bilinear')
        return out

    def flops(self, c, h, w):
        out = 5 * 5 * c * h * w
        return out


class pyrup(torch.nn.Module):
    def __init__(self, channels):
        super(pyrup, self).__init__()

        self.net = nn.Conv2d(channels, channels, kernel_size=5, groups=channels, padding=2, stride=1, bias=False)
        kernel = _get_pyramid_gaussian_kernel().unsqueeze(1).expand(channels, -1, -1, -1).clone()
        self.net.weight = torch.nn.Parameter(kernel)

    def forward(self, x):
        b, c, h, w = x.shape
        x_up = F.interpolate(x, size=(int(h * 2), int(w * 2)), mode='bilinear')
        out = self.net(x_up)
        return out

    def flops(self, c, h, w):
        out = 5 * 5 * c * (h * 2) * (w * 2)
        return out


class gauss_pyramid(torch.nn.Module):
    def __init__(self, scale):
        super(gauss_pyramid, self).__init__()

        self.pyrdowns = nn.ModuleList(nn.Sequential(pyrdown(C_all)) for _ in range(scale - 1))

    def forward(self, x, scale):
        temp = x.clone()
        pyramid_images = []
        pyramid_images.append(space_to_depth(temp, 2))
        for i in range(1, scale):
            out = self.pyrdowns[i - 1](temp)
            pyramid_images.append(space_to_depth(out, 2))
            temp = out
        return pyramid_images

    def flops(self, c, h, w, scale):
        out = 0
        nh = h
        nw = w
        for i in range(1, scale):
            out += self.pyrdowns[i - 1][0].flops(c, nh, nw)
            nh = nh // 2
            nw = nw // 2
        return out


class gauss_lapalian_pyramid(torch.nn.Module):
    def __init__(self, scale):
        super(gauss_lapalian_pyramid, self).__init__()

        self.scale = scale
        self.gauss_pyramid = gauss_pyramid(scale)
        self.pyrups = nn.ModuleList(nn.Sequential(pyrup(C_all)) for _ in range(scale - 1))

    def forward(self, x):
        gauss = self.gauss_pyramid(x, self.scale)

        lapalian = []
        for i in range(0, self.scale - 1):
            expand = self.pyrups[i](F.pixel_shuffle(gauss[i + 1], 2))
            lapalian.append(gauss[i] - space_to_depth(expand, 2))
        return gauss, lapalian

    def flops(self, c, h, w, scale):
        out = self.gauss_pyramid.flops(c, h, w, scale)
        nh = h // 2
        nw = w // 2
        for i in range(0, self.scale - 1):
            out += self.pyrups[i][0].flops(c, nh, nw)
            nh = nh // 2
            nw = nw // 2
        return out


class Pyramid_Collapse(torch.nn.Module):
    def __init__(self, scale):
        super(Pyramid_Collapse, self).__init__()

        self.scale = scale
        self.pyrups = nn.ModuleList(nn.Sequential(pyrup(C_all)) for _ in range(scale - 1))

    def forward(self, Gc, Lr):
        denoised_Gc = []
        scale = len(Gc)
        scale_l = len(Lr)

        expand = self.pyrups[0](F.pixel_shuffle(Gc[scale - 1], 2))  # 4
        denoised_Gc.append(expand + F.pixel_shuffle(Lr[scale_l - 1], 2))  # 4
        for l in range(1, scale - 1):
            expand = self.pyrups[l](denoised_Gc[l - 1])
            denoised_Gc.append(expand + F.pixel_shuffle(Lr[scale_l - 1 - l], 2))
        return denoised_Gc[scale_l - 1]

    def flops(self, c, h, w, scale):
        out = 0
        nh = h // 2
        nw = w // 2
        for i in range(scale - 1):
            out += self.pyrups[i][0].flops(c, nh, nw)
            nh = nh // 2
            nw = nw // 2
        return out


class Depthwise_separable_conv(nn.Module):
    def __init__(self, channels, channels_out, kernel_size=3, padding=1, bias=False):
        super(Depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels,
                                   bias=bias)
        self.pointwise = nn.Conv2d(channels, channels_out, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

    def flops(self, h, w, c_in, c_out):
        out = c_in * 3 * 3 * h * w + c_in * 1 * 1 * c_out * h * w
        return out


class Fusion_down(nn.Module):
    def __init__(self):
        super(Fusion_down, self).__init__()
        self.net1 = nn.Conv2d(17, 16, kernel_size=1, stride=1, padding=0)
        self.net2 = Depthwise_separable_conv(16, 16)
        self.net3 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        net1 = F.relu(self.net1(x))
        net2 = F.relu(self.net2(net1))
        out = F.sigmoid(self.net3(net2))
        return out

    def flops(self, h, w):
        out = (17 * 1 * 1 + 1) * 16 * h * w
        out += self.net2.flops(h, w, 16, 16)
        out += (16 * 1 * 1 + 1) * 1 * h * w
        return out


class Fusion_up(nn.Module):
    def __init__(self):
        super(Fusion_up, self).__init__()
        self.net1 = nn.Conv2d(18, 16, kernel_size=1, stride=1, padding=0)
        self.net2 = Depthwise_separable_conv(16, 16)
        self.net3 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        net1 = F.relu(self.net1(x))
        net2 = F.relu(self.net2(net1))
        out = F.sigmoid(self.net3(net2))
        return out

    def flops(self, h, w):
        out = (18 * 1 * 1 + 1) * 16 * h * w
        out += self.net2.flops(h, w, 16, 16)
        out += (16 * 1 * 1 + 1) * 1 * h * w
        return out


class Denoise_down(nn.Module):
    def __init__(self):
        super(Denoise_down, self).__init__()
        self.net1 = nn.Conv2d(17, 16, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        net1 = F.relu(self.net1(x))
        net2 = F.relu(self.net2(net1))
        out = self.net3(net2)
        return out

    def flops(self, h, w):
        out = (17 * 3 * 3 + 1) * 16 * h * w
        out += (16 * 3 * 3 + 1) * 16 * h * w
        out += (16 * 3 * 3 + 1) * 16 * h * w
        return out


class Denoise_up(nn.Module):
    def __init__(self):
        super(Denoise_up, self).__init__()
        self.net1 = nn.Conv2d(21, 16, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        net1 = F.relu(self.net1(x))
        net2 = F.relu(self.net2(net1))
        out = self.net3(net2)
        return out

    def flops(self, h, w):
        out = (21 * 3 * 3 + 1) * 16 * h * w
        out += (16 * 3 * 3 + 1) * 16 * h * w
        out += (16 * 3 * 3 + 1) * 16 * h * w
        return out


class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.net1 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0)
        self.net2 = Depthwise_separable_conv(16, 16)
        self.net3 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        net1 = F.relu(self.net1(x))
        net2 = F.relu(self.net2(net1))
        out = F.sigmoid(self.net3(net2))
        return out

    def flops(self, h, w):
        out = (32 * 1 * 1 + 1) * 16 * h * w
        out += self.net2.flops(h, w, 16, 16)
        out += (16 * 1 * 1 + 1) * 1 * h * w
        return out


class globalAttention(nn.Module):
    def __init__(self, num_feat=16, patch_size=4, heads=1):  # todo num_feat=64-->16, patch_size=8-->4
        super(globalAttention, self).__init__()
        self.heads = heads
        self.patch_size = patch_size
        self.dim = patch_size ** 2 * num_feat  # 16*16
        self.hidden_dim = self.dim // heads  # 16*16

        self.feat2patch = torch.nn.Unfold(kernel_size=patch_size, padding=0, stride=patch_size)

    def forward(self, m_in, m_out, q_in):
        b, t, c, h, w = m_in.shape  # B, 2 ,16,68,120
        H, D = self.heads, self.dim  # 1,16*16
        num_patch = (h // self.patch_size) * (w // self.patch_size)
        n, d = num_patch, self.hidden_dim  # 17*30 ,16*16

        q = q_in.unsqueeze(dim=1).view(-1, c, h, w)
        k = m_in.view(-1, c, h, w)
        v = m_out.view(-1, c, h, w)

        unfold_q = self.feat2patch(q)  # [B, 16*4*4,17*30]
        unfold_k = self.feat2patch(k)  # [B*2, 16*4*4,17*30]
        unfold_v = self.feat2patch(v)  # [B*2, 16*4*4,17*30]

        unfold_q = unfold_q.view(b, 1, H, d, n)  # [B, 1, H, 4*4*16, 17*30]
        unfold_k = unfold_k.view(b, t, H, d, n)  # [B, 2, H, 4*4*16, 17*30]
        unfold_v = unfold_v.view(b, t, H, d, n)  # [B, 2, H, 4*4*16, 17*30]

        unfold_q = unfold_q.permute(0, 2, 3, 1, 4).contiguous()  # [B, H, 4*4*16, 2, 17*30]
        unfold_k = unfold_k.permute(0, 2, 3, 1, 4).contiguous()  # [B, H, 4*4*16, 2, 17*30]
        unfold_v = unfold_v.permute(0, 2, 3, 1, 4).contiguous()  # [B, H, 4*4*16, 2, 17*30]

        unfold_q = unfold_q.view(b, H, d, n)  # [B, H, 16*16, 17*30]
        unfold_k = unfold_k.view(b, H, d, t * n)  # [B, H, 16*16, 2*17*30]
        unfold_v = unfold_v.view(b, H, d, t * n)  # [B, H, 16*16, 2*17*30]

        attn = torch.matmul(unfold_q.transpose(2, 3), unfold_k)  # [B, H, 17*30, 2*17*30]
        attn = attn * (d ** (-0.5))  # [B, H, 17*30, 2*17*30]
        attn = F.softmax(attn, dim=-1)  # [B, H, 17*30, 2*17*30]

        attn_x = torch.matmul(attn, unfold_v.transpose(2, 3))  # [B, H, 17*30, 16*16]
        attn_x = attn_x.view(b, H, 1, n, d)  # [B, H, 1, 17*30, 16*16]
        attn_x = attn_x.permute(0, 2, 1, 4, 3).contiguous()  # [B, 1, H, 16*16, 17*30]
        attn_x = attn_x.view(b, D, n)  # [B, 16*16, 17*30]
        patch2feat = torch.nn.Fold(output_size=(h, w), kernel_size=self.patch_size, padding=0, stride=self.patch_size)
        feat = patch2feat(attn_x)  # [B, 16, 68, 120]

        out = feat.view(q_in.shape)  # [B, 2, 16, 68, 120]
        # out += x  # [B, 2, 16, 68, 120]
        return out

    def flops(self, h, w, t):
        num_patch = (h // self.patch_size) * (w // self.patch_size)
        n, d = num_patch, self.hidden_dim

        out = 2 * d * n * (t * n)
        return out


class VideoDenoise(nn.Module):
    def __init__(self):
        super(VideoDenoise, self).__init__()

        self.fusion = Fusion_down()
        self.denoise = Denoise_down()
        self.refine = Refine()

        self.Memory = globalAttention()
        self.reduce_dim = nn.Conv2d(32, 16, kernel_size=3, padding=1)

    def forward(self, gp_ft0, lp_ft0, gp_ft1, lp_ft1, gp_pres, lp_pres, coeff_a, coeff_b):
        ll0 = gp_ft0  # ft0[:, 0:4, :, :]
        ll1 = gp_ft1  # ft1[:, 0:4, :, :]

        lp_ft1_s = self.Memory(gp_pres, lp_pres, gp_ft1)
        lp_ft1_m = self.reduce_dim(torch.cat([lp_ft1, lp_ft1_s], dim=1))

        # fusion
        sigma_ll1 = torch.mean(ll1, dim=1, keepdim=True) * coeff_a + coeff_b
        fusion_in = torch.cat([abs(ll1 - ll0), sigma_ll1], dim=1)
        filters = self.fusion(fusion_in)
        fusion_out = torch.mul(lp_ft0, (1 - filters)) + torch.mul(lp_ft1_m, filters)

        # denoise
        fusion_gp = torch.mul(ll0, (1 - filters)) + torch.mul(ll1, filters)
        sigma = torch.mean(fusion_gp, dim=1, keepdim=True) * coeff_a + coeff_b
        denoise_in = torch.cat([fusion_out, sigma], dim=1)
        denoise_out = self.denoise(denoise_in)

        # refine
        refine_in = torch.cat([fusion_out, denoise_out], axis=1)  # 1 * 33 * 64 * 64
        filters2 = self.refine(refine_in)  # 1 * 1 * 64 * 64
        refine_out = torch.mul(denoise_out, (1 - filters2)) + torch.mul(fusion_out, filters2)
        return filters, fusion_out, denoise_out, refine_out, filters2

    def flops(self, h, w):
        out = self.fusion.flops(h, w) + self.denoise.flops(h, w) + self.refine.flops(h, w)
        out += (32 * 3 * 3 + 1) * 16 * h * w
        out += self.Memory.flops(h, w, 3)
        return out


class MultiVideoDenoise(nn.Module):
    def __init__(self):
        super(MultiVideoDenoise, self).__init__()
        self.fusion = Fusion_up()
        self.denoise = Denoise_up()
        self.refine = Refine()

        self.Memory = globalAttention(patch_size=8)
        self.reduce_dim = nn.Conv2d(32, 16, kernel_size=3, padding=1)

    def forward(self, gp_ft0, lp_ft0, gp_ft1, lp_ft1, gp_pres, lp_pres, gamma_up, denoise_down, coeff_a, coeff_b):
        ll0 = gp_ft0  # ft0[:, 0:4, :, :]
        ll1 = gp_ft1  # ft1[:, 0:4, :, :]

        lp_ft1_s = self.Memory(gp_pres, lp_pres, gp_ft1)
        lp_ft1_m = self.reduce_dim(torch.cat([lp_ft1, lp_ft1_s], dim=1))

        # fusion
        sigma_ll1 = torch.mean(ll1, dim=1, keepdim=True) * coeff_a + coeff_b
        fusion_in = torch.cat([abs(ll1 - ll0), gamma_up, sigma_ll1], dim=1)
        filters = self.fusion(fusion_in)
        fusion_out = torch.mul(lp_ft0, (1 - filters)) + torch.mul(lp_ft1_m, filters)

        # denoise
        fusion_gp = torch.mul(ll0, (1 - filters)) + torch.mul(ll1, filters)
        sigma = torch.mean(fusion_gp, dim=1, keepdim=True) * coeff_a + coeff_b
        denoise_in = torch.cat([fusion_out, denoise_down, sigma], dim=1)
        denoise_out = self.denoise(denoise_in)

        # refine
        refine_in = torch.cat([fusion_out, denoise_out], axis=1)  # 1 * 33 * 64 * 64
        filters2 = self.refine(refine_in)  # 1 * 1 * 64 * 64
        refine_out = torch.mul(denoise_out, (1 - filters2)) + torch.mul(fusion_out, filters2)
        return filters, fusion_out, denoise_out, refine_out, filters2, lp_ft1_s

    def flops(self, h, w):
        out = self.fusion.flops(h, w) + self.denoise.flops(h, w) + self.refine.flops(h, w)
        out += (32 * 3 * 3 + 1) * 16 * h * w
        out += self.Memory.flops(h, w, 3)
        return out


class MultiVideoDenoise0(nn.Module):
    def __init__(self):
        super(MultiVideoDenoise0, self).__init__()
        self.fusion = Fusion_up()
        self.denoise = Denoise_up()
        self.refine = Refine()

        self.reduce_dim = nn.Conv2d(20, 16, kernel_size=1, padding=0)

    def forward(self, gp_ft0, lp_ft0, gp_ft1, lp_ft1, lp_ft1_s, gamma_up, denoise_down, coeff_a, coeff_b):
        ll0 = gp_ft0  # ft0[:, 0:4, :, :]
        ll1 = gp_ft1  # ft1[:, 0:4, :, :]

        lp_ft1_m = self.reduce_dim(torch.cat([lp_ft1, lp_ft1_s], dim=1))

        # fusion
        sigma_ll1 = torch.mean(ll1, dim=1, keepdim=True) * coeff_a + coeff_b
        fusion_in = torch.cat([abs(ll1 - ll0), gamma_up, sigma_ll1], dim=1)
        filters = self.fusion(fusion_in)
        fusion_out = torch.mul(lp_ft0, (1 - filters)) + torch.mul(lp_ft1_m, filters)

        # denoise
        fusion_gp = torch.mul(ll0, (1 - filters)) + torch.mul(ll1, filters)
        sigma = torch.mean(fusion_gp, dim=1, keepdim=True) * coeff_a + coeff_b
        denoise_in = torch.cat([fusion_out, denoise_down, sigma], dim=1)
        denoise_out = self.denoise(denoise_in)

        # refine
        refine_in = torch.cat([fusion_out, denoise_out], axis=1)  # 1 * 33 * 64 * 64
        filters2 = self.refine(refine_in)  # 1 * 1 * 64 * 64
        refine_out = torch.mul(denoise_out, (1 - filters2)) + torch.mul(fusion_out, filters2)
        return filters, fusion_out, denoise_out, refine_out, filters2

    def flops(self, h, w):
        out = self.fusion.flops(h, w) + self.denoise.flops(h, w) + self.refine.flops(h, w)
        out += (20 * 1 * 1 + 1) * 16 * h * w
        return out


class MainDenoise(nn.Module):
    def __init__(self, scale=4):
        super(MainDenoise, self).__init__()

        self.vd = VideoDenoise()
        self.md1 = MultiVideoDenoise()
        self.md0 = MultiVideoDenoise0()

        self.gauss_lapalian_pyramid = gauss_lapalian_pyramid(scale)
        self.Pyramid_Collapse = Pyramid_Collapse(scale)

    def forward(self, x_pres, x, coeff_a=1, coeff_b=1):
        x_3 = x_pres[0]
        x_2 = x_pres[1]
        x_1 = x_pres[2]

        ft0 = x[:, 0:4, :, :]  # 1*4*128*128, the t-1 fusion frame
        ft1 = x[:, 4:8, :, :]  # 1*4*128*128, the t frame

        gp_x_3, lp_x_3 = self.gauss_lapalian_pyramid(x_3)
        gp_x_2, lp_x_2 = self.gauss_lapalian_pyramid(x_2)
        gp_x_1, lp_x_1 = self.gauss_lapalian_pyramid(x_1)

        gp_t0, lp_t0 = self.gauss_lapalian_pyramid(ft0)
        gp_t1, lp_t1 = self.gauss_lapalian_pyramid(ft1)

        lp_t1_fusion, lp_t1_denoise, lp_t1_refine = [], [], []
        filters_all, filters2_all = [], []

        filters, fusion_out, denoise_out, refine_out, filters2 = \
            self.vd(gp_t0[2], lp_t0[2], gp_t1[2], lp_t1[2],
                    torch.stack([gp_x_3[2], gp_x_2[2], gp_x_1[2]], dim=1),
                    torch.stack([lp_x_3[2], lp_x_2[2], lp_x_1[2]], dim=1),
                    coeff_a, coeff_b)  # [1, 16, 16, 16]

        lp_t1_fusion.append(fusion_out)
        lp_t1_denoise.append(denoise_out)
        lp_t1_refine.append(refine_out)
        filters_all.append(filters)
        filters2_all.append(filters2)

        denoise_out_d2 = F.pixel_shuffle(denoise_out, 2)  # [1, 4, 32, 32]
        filters_up_d2 = F.upsample(filters, scale_factor=2)

        filters, fusion_out, denoise_out, refine_out, filters2, lp_ft1_s = \
            self.md1(gp_t0[1], lp_t0[1], gp_t1[1], lp_t1[1],
                     torch.stack([gp_x_3[1], gp_x_2[1], gp_x_1[1]], dim=1),
                     torch.stack([lp_x_3[1], lp_x_2[1], lp_x_1[1]], dim=1),
                     filters_up_d2, denoise_out_d2, coeff_a, coeff_b)  # [1, 16, 32, 32]
        lp_t1_fusion.append(fusion_out)
        lp_t1_denoise.append(denoise_out)
        lp_t1_refine.append(refine_out)

        filters_all.append(filters)
        filters2_all.append(filters2)

        denoise_up_d1 = F.pixel_shuffle(denoise_out, 2)  # [1, 4, 64, 64]
        filters_up_d1 = F.upsample(filters, scale_factor=2)

        filters, fusion_out, denoise_out, refine_out, filters2, = \
            self.md0(gp_t0[0], lp_t0[0], gp_t1[0], lp_t1[0],
                     F.pixel_shuffle(lp_ft1_s, 2),
                     filters_up_d1, denoise_up_d1, coeff_a, coeff_b)
        lp_t1_fusion.append(fusion_out)
        lp_t1_denoise.append(denoise_out)
        lp_t1_refine.append(refine_out)
        filters_all.append(filters)
        filters2_all.append(filters2)
        # [1, 16, 64, 64]

        lp_t1_fusion = lp_t1_fusion[::-1]
        lp_t1_denoise = lp_t1_denoise[::-1]
        lp_t1_refine = lp_t1_refine[::-1]

        fusion_out = self.Pyramid_Collapse(gp_t1, lp_t1_fusion)  # [1, 4, 128, 128]
        denoise_out = self.Pyramid_Collapse(gp_t1, lp_t1_denoise)
        refine_out = self.Pyramid_Collapse(gp_t1, lp_t1_refine)
        return filters, fusion_out, denoise_out, filters2, refine_out

    def flops(self, b, c, h, w, scale=4):
        out = self.vd.flops(h // 8, w // 8) + self.md1.flops(h // 4, w // 4) + self.md0.flops(h // 2, w // 2) \
              + self.gauss_lapalian_pyramid.flops(c, h, w, scale) * 5 \
              + self.Pyramid_Collapse.flops(c, h, w, scale) * 2
        return out * b


if __name__ == '__main__':
    from torch.autograd import Variable

    net = MainDenoise().cuda()
    # print(net)
    # input = Variable(torch.FloatTensor(1, 8, 128, 128)).cuda()
    # pres = [Variable(torch.FloatTensor(1, 4, 128, 128)).cuda(),
    #         Variable(torch.FloatTensor(1, 4, 128, 128)).cuda(),
    #         Variable(torch.FloatTensor(1, 4, 128, 128)).cuda()]
    input = Variable(torch.FloatTensor(1, 8, 544, 960)).cuda()
    pres = [Variable(torch.FloatTensor(1, 4, 544, 960)).cuda(),
            Variable(torch.FloatTensor(1, 4, 544, 960)).cuda(),
            Variable(torch.FloatTensor(1, 4, 544, 960)).cuda()]
    y = net(pres, input)
    print(y[-1].shape)
    print(net.flops(1, 4, 544, 960) / 1e9)

    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

    from thop import profile

    flops, params = profile(net, inputs=(pres, input,))
    print("%.6f GFLOPs." % (flops / 1e9))
    print("%.6f Params(M)." % (params / 1e6))
