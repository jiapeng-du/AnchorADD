import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch import Tensor

___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"


# =========================
# SincConv: 单通道前端滤波器（基本保持原始实现）
# =========================
class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, device, out_channels, kernel_size,
                 in_channels=1, sample_rate=16000,
                 stride=1, padding=0, dilation=1, bias=False, groups=1):

        super(SincConv, self).__init__()

        if in_channels != 1:
            msg = "SincConv only supports one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetric)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        # initialize filterbanks using Mel scale
        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)  # Hz to mel conversion
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)  # Mel to Hz conversion
        self.mel = filbandwidthsf

        self.hsupp = torch.arange(
            -(self.kernel_size - 1) / 2,
            (self.kernel_size - 1) / 2 + 1
        )
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)

    def forward(self, x):
        # x: [B, 1, T]
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(
                2 * fmax * self.hsupp / self.sample_rate
            )
            hLow = (2 * fmin / self.sample_rate) * np.sinc(
                2 * fmin * self.hsupp / self.sample_rate
            )
            hideal = hHigh - hLow
            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hideal)

        band_pass_filter = self.band_pass.to(self.device)
        self.filters = band_pass_filter.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(
            x,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


# =========================
# 残差块（基本保持原始实现）
# =========================
class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(Residual_block, self).__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features=nb_filts[0])

        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        self.conv1 = nn.Conv1d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=3,
            padding=1,
            stride=1,
        )

        self.bn2 = nn.BatchNorm1d(num_features=nb_filts[1])
        self.conv2 = nn.Conv1d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            padding=1,
            kernel_size=3,
            stride=1,
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=0,
                kernel_size=1,
                stride=1,
            )
        else:
            self.downsample = False

        self.mp = nn.MaxPool1d(3)

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


# =========================
# RawNet + 真正的 ∆−UQ 实现
# =========================
class RawNet(nn.Module):
    def __init__(self, d_args, device):
        """
        d_args 结构沿用原始 RawNet：
          - 'filts': [ C_front, [c0_in, c0_out], [c1_in, c1_out] ]
          - 'first_conv', 'in_channels', 'gru_node', 'nb_gru_layer',
            'nb_fc_node', 'nb_classes'
          - 新增: 'delta_uq' (bool), 'uq_K' (int)
        """
        super(RawNet, self).__init__()
        self.device = device

        # ==== Delta-UQ 修改1：新增开关与超参 ====
        self.delta_uq = bool(d_args.get("delta_uq", False))  # 是否启用 ∆−UQ
        self.uq_K = int(d_args.get("uq_K", 32))              # UQ 时 anchor 采样次数

        # 前端通道数（SincConv 输出）
        self.front_channels = d_args["filts"][0]

        # ========= 前端 SincConv =========
        self.Sinc_conv = SincConv(
            device=self.device,
            out_channels=d_args["filts"][0],
            kernel_size=d_args["first_conv"],
            in_channels=d_args["in_channels"],
        )

        self.first_bn = nn.BatchNorm1d(num_features=d_args["filts"][0])
        self.selu = nn.SELU(inplace=True)

        # ========= 残差块 + 注意力 =========
        # 原始配置
        c0_in_orig, c0_out = d_args["filts"][1]

        # ==== Delta-UQ 修改2：block0 / block1 的通道配置 ====
        if self.delta_uq:
            # block0: 输入是 [F(c), F(x)-F(c)]，通道 = 2 * front_channels
            filts_block0 = [2 * self.front_channels, c0_out]
            # block1: 输入就是 block0 输出的 c0_out 通道
            filts_block1 = [c0_out, c0_out]
        else:
            # 不用 ∆−UQ 时，保持和原始 RawNet 完全一致
            filts_block0 = [c0_in_orig, c0_out]
            filts_block1 = [c0_in_orig, c0_out]

        self.block0 = nn.Sequential(
            Residual_block(nb_filts=filts_block0, first=True)
        )
        self.block1 = nn.Sequential(
            Residual_block(nb_filts=filts_block1)
        )

        # block2-5 保持和原实现一致
        self.block2 = nn.Sequential(Residual_block(nb_filts=d_args["filts"][2]))
        d_args["filts"][2][0] = d_args["filts"][2][1]
        self.block3 = nn.Sequential(Residual_block(nb_filts=d_args["filts"][2]))
        self.block4 = nn.Sequential(Residual_block(nb_filts=d_args["filts"][2]))
        self.block5 = nn.Sequential(Residual_block(nb_filts=d_args["filts"][2]))

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc_attention0 = self._make_attention_fc(
            in_features=filts_block0[1],
            l_out_features=filts_block0[1],
        )
        self.fc_attention1 = self._make_attention_fc(
            in_features=filts_block1[1],
            l_out_features=filts_block1[1],
        )
        self.fc_attention2 = self._make_attention_fc(
            in_features=d_args["filts"][2][-1],
            l_out_features=d_args["filts"][2][-1],
        )
        self.fc_attention3 = self._make_attention_fc(
            in_features=d_args["filts"][2][-1],
            l_out_features=d_args["filts"][2][-1],
        )
        self.fc_attention4 = self._make_attention_fc(
            in_features=d_args["filts"][2][-1],
            l_out_features=d_args["filts"][2][-1],
        )
        self.fc_attention5 = self._make_attention_fc(
            in_features=d_args["filts"][2][-1],
            l_out_features=d_args["filts"][2][-1],
        )

        # ========= GRU + 全连接 =========
        self.bn_before_gru = nn.BatchNorm1d(num_features=d_args["filts"][2][-1])
        self.gru = nn.GRU(
            input_size=d_args["filts"][2][-1],
            hidden_size=d_args["gru_node"],
            num_layers=d_args["nb_gru_layer"],
            batch_first=True,
        )

        self.fc1_gru = nn.Linear(
            in_features=d_args["gru_node"], out_features=d_args["nb_fc_node"]
        )
        self.fc2_gru = nn.Linear(
            in_features=d_args["nb_fc_node"],
            out_features=d_args["nb_classes"],
            bias=True,
        )

        self.logsoftmax = nn.LogSoftmax(dim=1)

    # ========= 前端：SincConv + BN + SELU =========
    def _front_end(self, x_wave):
        """
        x_wave: [B, T]
        返回特征: [B, C, L]
        """
        B, T = x_wave.shape
        x = x_wave.view(B, 1, T)  # [B,1,T]
        x = self.Sinc_conv(x)     # [B,C,L']
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x = self.selu(x)
        return x

    # ========= 后端：残差块 + 注意力 + GRU + FC =========
    def _back_end(self, x):
        # block0
        x0 = self.block0(x)
        y0 = self.avgpool(x0).view(x0.size(0), -1)
        y0 = self.fc_attention0(y0)
        y0 = torch.sigmoid(y0).view(y0.size(0), y0.size(1), 1)
        x = x0 * y0 + y0

        # block1
        x1 = self.block1(x)
        y1 = self.avgpool(x1).view(x1.size(0), -1)
        y1 = self.fc_attention1(y1)
        y1 = torch.sigmoid(y1).view(y1.size(0), y1.size(1), 1)
        x = x1 * y1 + y1

        # block2
        x2 = self.block2(x)
        y2 = self.avgpool(x2).view(x2.size(0), -1)
        y2 = self.fc_attention2(y2)
        y2 = torch.sigmoid(y2).view(y2.size(0), y2.size(1), 1)
        x = x2 * y2 + y2

        # block3
        x3 = self.block3(x)
        y3 = self.avgpool(x3).view(x3.size(0), -1)
        y3 = self.fc_attention3(y3)
        y3 = torch.sigmoid(y3).view(y3.size(0), y3.size(1), 1)
        x = x3 * y3 + y3

        # block4
        x4 = self.block4(x)
        y4 = self.avgpool(x4).view(x4.size(0), -1)
        y4 = self.fc_attention4(y4)
        y4 = torch.sigmoid(y4).view(y4.size(0), y4.size(1), 1)
        x = x4 * y4 + y4

        # block5
        x5 = self.block5(x)
        y5 = self.avgpool(x5).view(x5.size(0), -1)
        y5 = self.fc_attention5(y5)
        y5 = torch.sigmoid(y5).view(y5.size(0), y5.size(1), 1)
        x = x5 * y5 + y5

        # GRU + FC
        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = x.permute(0, 2, 1)  # [B, time, feat]
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1_gru(x)
        x = self.fc2_gru(x)
        return self.logsoftmax(x)

    # ========= forward：单次随机 anchor（训练 & 普通推理） =========
    def forward(self, x, train=False):
        """
        x: [B, T] waveform
        若 self.delta_uq=True：
          - 随机 batch 内 anchor c
          - 在前端特征上构造 [F(c), F(x)-F(c)]
        """
        B, T = x.shape

        if self.delta_uq:
            # ==== Delta-UQ 修改3：在前端特征空间做 [F(c), F(x)-F(c)] ====
            # 在 batch 内随机采样 anchor
            idx = torch.randperm(B, device=x.device)
            anchor_wave = x[idx]  # [B,T]

            # 一次性通过前端，保证 BN 统计一致
            stacked = torch.cat([x, anchor_wave], dim=0)  # [2B, T]
            feats = self._front_end(stacked)              # [2B, C, L]
            x_feat, c_feat = feats[:B], feats[B:]         # F(x), F(c)

            # Reparameterization: [F(c), F(x)-F(c)]
            x_repr = torch.cat([c_feat, x_feat - c_feat], dim=1)  # [B,2C,L]
        else:
            x_repr = self._front_end(x)  # [B,C,L]

        out = self._back_end(x_repr)
        return out

    # ========= 多 anchor 推理：真正的不确定性估计 =========
    def predict_with_uq(self, x):
        """
        x: [B,T]
        返回:
            mu:    [B,C] K 次 logits 的均值
            sigma: [B,C] K 次 logits 的标准差
        """
        # ==== Delta-UQ 修改4：多次随机 anchor 前向，统计 μ/σ ====
        preds = []
        for _ in range(self.uq_K):
            with torch.no_grad():
                logits = self.forward(x, train=False)
            preds.append(logits)

        preds = torch.stack(preds, dim=0)  # [K,B,C]
        mu = preds.mean(dim=0)
        sigma = preds.std(dim=0)
        return sigma

    # ========= 辅助 FC =========
    def _make_attention_fc(self, in_features, l_out_features):
        return nn.Sequential(nn.Linear(in_features=in_features,
                                       out_features=l_out_features))

    # ========= （可选）summary，方便调试 =========
    def summary(self, input_size, batch_size=-1, device="cuda", print_fn=None):
        if print_fn is None:
            print_fn = print
        model = self

        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)
                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    if len(summary[m_key]["output_shape"]) != 0:
                        summary[m_key]["output_shape"][0] = batch_size

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

            if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
            ):
                hooks.append(module.register_forward_hook(hook))

        device = device.lower()
        assert device in ["cuda", "cpu"], "device must be 'cuda' or 'cpu'"

        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        if isinstance(input_size, tuple):
            input_size = [input_size]
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
        summary = OrderedDict()
        hooks = []
        model.apply(register_hook)
        model(*x)
        for h in hooks:
            h.remove()

        print_fn("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format(
            "Layer (type)", "Output Shape", "Param #"
        )
        print_fn(line_new)
        print_fn("================================================================")
        total_params = 0
        trainable_params = 0
        for layer in summary:
            line_new = "{:>20}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
            total_params += summary[layer]["nb_params"]
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] is True:
                    trainable_params += summary[layer]["nb_params"]
            print_fn(line_new)
        print_fn("================================================================")
        print_fn("Total params: {0:,}".format(total_params))
        print_fn("Trainable params: {0:,}".format(trainable_params))