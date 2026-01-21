#import sys
#sys.path.append("./")
#
#import torch
#import torch.nn as nn
#from transformers import Wav2Vec2Model, AutoConfig
#
#from utils.ideas.MoEF.aasist import W2VAASIST
#from utils.ideas.MoEF.moef import MoE24fusion
#
#
## ========== 参数配置 ==========
#class this_arg():
#    stage = 1
#    moe_topk = 2
#    moe_experts = 4
#    moe_exp_hid = 128
#
#    ### ==== 新增：Delta-UQ 开关与参数 ====
#    delta_uq = True
#    uq_K = 10         # 推理时 anchor 采样次数
#
#
## ========== 主模型 ==========
#class Model(nn.Module):
#    def __init__(self, args=this_arg()):
#        super().__init__()
#        self.args = args
#
#        ### ==== 修改1：让 Wav2Vec2 接受 2 通道输入 ====
#        config = AutoConfig.from_pretrained(
#            "./data/pretrained_model/facebook/wav2vec2-xls-r-300m"
#        )
#        config.num_channels = 2  # ★★ 关键：W2V2 front-end CNN 支持 2 通道输入 ★★
#
#        self.pretrain_model = Wav2Vec2Model.from_pretrained(
#            "./data/pretrained_model/facebook/wav2vec2-xls-r-300m",
#            config=config,
#            local_files_only=True
#        )
#
#        # 下游结构保持原样
#        self.classifier = W2VAASIST()
#        self.moe_l = MoE24fusion(
#            ds_inputsize=1024,
#            input_size=1024,
#            output_size=1024,
#            num_experts=24 * args.moe_experts,
#            hidden_size=args.moe_exp_hid,
#            noisy_gating=True,
#            k=args.moe_topk,
#            trainingmode=True
#        )
#
#        # 冻结 W2V2
#        for p in self.pretrain_model.parameters():
#            p.requires_grad = False
#
#
#    # ========== (核心) 构造 [anchor, x-anchor] 输入 ==========
##    def _build_two_channel(self, x):
##        """
##        x: [B, T]
##        返回:
##            inp: [B, 2, T]
##        """
##        B = x.size(0)
##
##        # 随机采样 anchor
##        idx = torch.randperm(B, device=x.device)
##        anchor = x[idx]
##
##        # x, anchor 均为 [B,T]，所以只需要 unsqueeze 一次
##        anchor_ch = anchor.unsqueeze(1)        # [B,1,T]
##        delta_ch  = (x - anchor).unsqueeze(1)  # [B,1,T]
##
##        return torch.cat([anchor_ch, delta_ch], dim=1)   # [B,2,T]
##
##
##
##    # ========== 单次 forward（训练 or 单次推理） ==========
##    def forward(self, x, train=False):
##        """
##        x: [B,T] waveform
##        """
##        print(x.shape)
##	    # ==========================        
##	    
##        if self.args.delta_uq and train:
##            ### ==== 修改2：训练时启用真正的 ∆−UQ 两通道 ====
##            x = self._build_two_channel(x)
##        else:
##            ### ==== 修改3：推理时也保持两通道结构，但 anchor=0 ====
##            zero_anchor = torch.zeros_like(x).unsqueeze(1)   # [B,1,T]
##            print(zero_anchor.shape)            
##            x_ch = x.unsqueeze(1)                            # [B,1,T]
##            print(x_ch.shape)            
##            x = torch.cat([zero_anchor, x_ch], dim=1)        # [B,2,T]
##            print(x.shape)
##
##
##            
##        # W2V2 前向
##        with torch.no_grad():
##            feats = self.pretrain_model(
##                x,
##                output_hidden_states=True
##            )
##
##        B, T, D = feats.last_hidden_state.shape
##
##        # MoE 输入准备
##        hidden_ones = [feats.hidden_states[i].reshape(B*T, D) for i in range(24)]
##        moe_input = feats.last_hidden_state.reshape(B*T, D)
##
##        # MoE
##        fusion_x = self.moe_l(
##            moe_input,
##            hidden_ones,
##            training=train
##        )
##
##        # 分类器
##        pred, hidden_state = self.classifier(
##            fusion_x[0].reshape(B, T, D)
##        )
##        return pred
#
#
#    def _build_two_channel(self, x):
#	    """
#	    x: [B, T] 或 [B, 1, T]
#	    返回: [B, 2, T]
#	    """
#	    # 确保输入是 [B, T] 格式
#	    if x.dim() == 3:
#	        x = x.squeeze(1)  # 从 [B, 1, T] 变为 [B, T]
#	    
#	    B = x.size(0)
#	
#	    # 随机采样 anchor
#	    idx = torch.randperm(B)
#	    anchor = x[idx]
#	
#	    # 通道1: anchor
#	    anchor_ch = anchor.unsqueeze(1)              # [B, 1, T]
#	
#	    # 通道2: x-anchor  
#	    delta_ch = (x - anchor).unsqueeze(1)         # [B, 1, T]
#	
#	    two_ch = torch.cat([anchor_ch, delta_ch], dim=1)  # [B, 2, T]
#	    return two_ch
#	
#    def forward(self, x, train=False):
#	    """
#	    x: [B, T] 或 [B, 1, T] waveform
#	    """
#	    # 确保输入是 [B, T] 格式
#	    if x.dim() == 3:
#	        x = x.squeeze(1)
#	    
#	    if self.args.delta_uq and train:
#	        x = self._build_two_channel(x)
#	    else:
#	        # 推理时使用零anchor
#	        zero_anchor = torch.zeros_like(x).unsqueeze(1)   # [B, 1, T]
#	        x_ch = x.unsqueeze(1)                            # [B, 1, T]
#	        x = torch.cat([zero_anchor, x_ch], dim=1)        # [B, 2, T]
#	
#	    # 确保输入是3D: [B, 2, T]
#	    print(f"Input shape to Wav2Vec2: {x.shape}")  # 调试用
#	    
#	    # W2V2 前向
#	    with torch.no_grad():
#	        feats = self.pretrain_model(
#	            x,
#	            output_hidden_states=True
#	        )
#	    
#	    # 其余代码保持不变...
#	    B, T, D = feats.last_hidden_state.shape
#	
#	    # MoE 输入准备
#	    hidden_ones = [feats.hidden_states[i].reshape(B*T, D) for i in range(24)]
#	    moe_input = feats.last_hidden_state.reshape(B*T, D)
#	
#	    # MoE
#	    fusion_x = self.moe_l(
#	        moe_input,
#	        hidden_ones,
#	        training=train
#	    )
#	
#	    # 分类器
#	    pred, hidden_state = self.classifier(
#	        fusion_x[0].reshape(B, T, D)
#	    )
#	    return pred
#
#    # ========== (核心) ∆−UQ 多 anchor 推理：返回均值 μ + 方差 σ ==========
#    def predict_with_uq(self, x):
#        """
#        x: [B,T]
#        返回：
#            mu:    [B,C]  K 次预测的均值
#            sigma: [B,C]  K 次预测的标准差
#        """
#        K = self.args.uq_K
#        preds = []
#
#        for _ in range(K):
#            # 每次都换不同 anchor
#            x_uq = self._build_two_channel(x)
#
#            with torch.no_grad():
#                pred = self.forward(x_uq, train=False)
#
#            preds.append(pred)
#
#        preds = torch.stack(preds, dim=0)   # [K,B,C]
#
#        mu = preds.mean(dim=0)
#        sigma = preds.std(dim=0)
#
#        return mu, sigma
#
#
## ========== 最小测试 ==========
#if __name__ == "__main__":
#    args = this_arg()
#    model = Model(args)
#
#    x = torch.randn(2, 64000)  # 两条音频
#
#    print("=== 训练模式 ===")
#    out = model(x, train=True)
#    print(out.shape)
#
#    print("=== UQ 推理 ===")
#    mu, sigma = model.predict_with_uq(x)
#    print(mu.shape, sigma.shape)
import sys
sys.path.append("./")

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

from utils.ideas.MoEF.aasist import W2VAASIST
from utils.ideas.MoEF.moef import MoE24fusion


# ========== 参数配置 ==========
class this_arg:
    stage = 1
    moe_topk = 2
    moe_experts = 4
    moe_exp_hid = 128

    # Delta-UQ 开关与参数
    delta_uq =True    # 是否启用 ∆−UQ
    uq_K = 10         # 推理时 anchor 采样次数


# ========== 主模型 ==========
class Model(nn.Module):
    def __init__(self, args=this_arg()):
        super().__init__()
        self.args = args

        # 1) 正常加载 Wav2Vec2（不改通道数）
        self.pretrain_model = Wav2Vec2Model.from_pretrained(
            "./data/pretrained_model/facebook/wav2vec2-xls-r-300m",
            local_files_only=True
        )
        hidden_size = self.pretrain_model.config.hidden_size  # 通常是 1024

        # 2) Delta-UQ 特征投影层： [F(c), F(x)-F(c)] (2D) -> D
        self.use_delta_uq = bool(getattr(args, "delta_uq", False))
        if self.use_delta_uq:
            self.delta_proj = nn.Linear(2 * hidden_size, hidden_size)

        # 3) MoE & 分类器：保持原来 1024 维接口
        self.classifier = W2VAASIST()
        self.moe_l = MoE24fusion(
            ds_inputsize=hidden_size,
            input_size=hidden_size,
            output_size=hidden_size,
            num_experts=24 * args.moe_experts,
            hidden_size=args.moe_exp_hid,
            noisy_gating=True,
            k=args.moe_topk,
            trainingmode=True
        )

        # 冻结 W2V2
        for p in self.pretrain_model.parameters():
            p.requires_grad = False

    # ========== 一次性提取 W2V2 特征 ==========
    def _extract_feats(self, x):
        """
        x: [B, T]
        返回:
            last: [B, L, D]
            hidden_states: tuple(len=24/25) 每个 [B, L, D]
        """
        with torch.no_grad():
            out = self.pretrain_model(
                x,
                output_hidden_states=True,
                output_attentions=True
            )
        return out.last_hidden_state, out.hidden_states

    # ========== 单次 forward（训练 or 单次推理） ==========
    def forward(self, x, train=False):
        """
        x: [B,T] waveform
        返回:
            pred: [B, num_classes]
        """
        B, T = x.shape

        if self.use_delta_uq:
	            # ====== ∆−UQ：在特征空间做 [F(c), F(x)-F(c)] ======
            if train:     
	            # 1) 随机选择 anchor
                idx = torch.randperm(B, device=x.device)
                anchor = x[idx]  # [B,T]
            else:
                anchor = x 
                
            # 2) 一次性提特征，保证统计一致
            stacked = torch.cat([x, anchor], dim=0)   # [2B,T]
            last, hidden_states = self._extract_feats(stacked)  # [2B,L,D]
	
            x_last = last[:B]     # [B,L,D]
            a_last = last[B:]     # [B,L,D]
	
            # 3) 拼接 [F(c), F(x)-F(c)] -> [B,L,2D]
            two = torch.cat([a_last, x_last - a_last], dim=-1)
	
            # 4) 线性投影回 D 维
            last_feat = self.delta_proj(two)          # [B,L,D]
	
            # 只取 x 部分的 hidden_states 给 MoE 用
            hs_for_moe = [h[:B] for h in hidden_states[:24]]  # 每个 [B,L,D]
        else:
            # 不用 ∆−UQ：正常提特征
            last_feat, hidden_states = self._extract_feats(x)  # [B,L,D]
            hs_for_moe = [h for h in hidden_states[:24]]

        # ===== MoE 输入 reshape =====
        B, L, D = last_feat.shape
        moe_input = last_feat.reshape(B * L, D)               # [B*L, D]
        hidden_ones = [h.reshape(B * L, D) for h in hs_for_moe]

        # ===== MoE 前向 =====
        fusion_x = self.moe_l(
            moe_input,
            hidden_ones,
            training=train
        )  # fusion_x[0]: [B*L, D]

        # 还原成 [B,L,D] 给 AASIST
        fused_seq = fusion_x[0].reshape(B, L, D)

        # ===== 分类器 =====
        pred, hidden_state = self.classifier(fused_seq)  # pred: [B, num_classes]
        return pred

    # ========== 多 anchor 推理：返回 μ + σ ==========
    def predict_with_uq(self, x):
        """
        x: [B,T]
        返回：
            mu:    [B,C]  K 次预测的均值
            sigma: [B,C]  K 次预测的标准差
        """
        assert self.use_delta_uq, "predict_with_uq 只有在 delta_uq=True 时才有意义"

        K = getattr(self.args, "uq_K", 10)
        preds = []
        for _ in range(K):
            with torch.no_grad():
                # 每次 forward 里都会随机一个新的 anchor
                pred = self.forward(x, train=False)  # [B,C]
            preds.append(pred)

        preds = torch.stack(preds, dim=0)  # [K,B,C]
        mu = preds.mean(dim=0)
        sigma = preds.std(dim=0)
        return mu, sigma


# ========== 最小测试 ==========
if __name__ == "__main__":
    args = this_arg()
    model = Model(args)

    x = torch.randn(2, 64000)  # 两条音频

    print("=== 训练模式 ===")
    out = model(x, train=True)
    print("logits:", out.shape)

    print("=== UQ 推理 ===")
    mu, sigma = model.predict_with_uq(x)
    print("mu:", mu.shape, "sigma:", sigma.shape)