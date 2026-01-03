# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

# # 用于替换 LabelEmbedder 的新类，处理连续标签（实部和虚部）
# class ContinuousEmbedder(nn.Module):
#     def __init__(self, input_dim, hidden_size, dropout_prob):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, hidden_size),
#             nn.SiLU(),
#             nn.Linear(hidden_size, hidden_size),
#         )
#         # 如果需要 CFG (Classifier-Free Guidance)，可以在这里实现 dropout 逻辑
#         # 简单起见，训练时以 dropout_prob 的概率将输入置为 0 (代表无条件)

#     def forward(self, labels, train, force_drop_ids=None):
#         # labels: [Batch, 2] (实部, 虚部)
#         embeddings = self.mlp(labels)
#         return embeddings

# [修改] 替换原来的 ContinuousEmbedder 类
class ContinuousEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_size, dropout_prob):
        super().__init__()
        self.dropout_prob = dropout_prob  # 记录 dropout 概率
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, labels, train, force_drop_ids=None):
        # 1. 判断是否需要执行 Dropout
        use_dropout = self.dropout_prob > 0
        
        if (train and use_dropout) or (force_drop_ids is not None):
            if force_drop_ids is None:
                # 训练时：以 dropout_prob 的概率随机生成 mask
                drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            else:
                # 推理时：强制指定哪些需要丢弃
                drop_ids = force_drop_ids == 1
            
            # 2. 执行 Mask：将需要丢弃的标签设为全 0
            # unsqueeze(1) 是为了广播到 (Batch, 2)
            # labels = torch.where(drop_ids.unsqueeze(1), torch.zeros_like(labels), labels)

            # ---------------- 修改开始 ----------------
            # 2. 执行 Mask：将需要丢弃的标签设为 [2.0, 2.0]
            # 这种“不可能出现的物理参数”会被模型识别为无条件信号
            null_token = torch.tensor([2.0, 2.0], device=labels.device, dtype=labels.dtype)
            
            # 使用 torch.where 进行替换
            # drop_ids.unsqueeze(1) 形状是 [B, 1]，null_token 会自动广播到 [B, 2]
            labels = torch.where(drop_ids.unsqueeze(1), null_token, labels)
            # ---------------- 修改结束 ----------------

        # 3. 经过 MLP 得到 Embedding
        embeddings = self.mlp(labels)
        return embeddings

# 1. [新增] LearnableContinuousEmbedder 类
class LearnableContinuousEmbedder(nn.Module):
    """
    使用独立的可学习参数 (nn.Parameter) 作为 Null Embedding，
    而不是将 [2, 2] 输入 MLP。
    """
    def __init__(self, input_dim, hidden_size, dropout_prob):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # [核心] 这是一个独立的向量，不依赖 MLP
        self.null_embedding = nn.Parameter(torch.randn(1, hidden_size))

    def forward(self, labels, train, force_drop_ids=None):
        # A. 先计算所有物理条件的 Embedding
        embeddings = self.mlp(labels)

        # B. 计算 Dropout Mask
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            if force_drop_ids is None:
                drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            else:
                drop_ids = force_drop_ids == 1
            
            # C. [替换逻辑] 直接用 null_embedding 替换掉对应的行
            if drop_ids.any():
                embeddings = torch.where(
                    drop_ids.unsqueeze(1), 
                    self.null_embedding.to(embeddings.dtype), 
                    embeddings
                )
        return embeddings

class FourierContinuousEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_size, dropout_prob, freq_embedding_size=256):
        super().__init__()
        # input_dim=2 (实部, 虚部)
        # 我们为每个维度单独生成 freq_embedding_size 的正弦特征
        # 所以 MLP 的输入维度是 input_dim * freq_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * freq_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.dropout_prob = dropout_prob
        self.freq_embedding_size = freq_embedding_size
        
        # 定义一个可学习的 Null Embedding (保留你之前的 Learnable 方案)
        self.null_embedding = nn.Parameter(torch.randn(1, hidden_size))

    def sinusoidal_embedding(self, x):
        # x: [N, 2]
        # 我们可以复用 TimestepEmbedder.timestep_embedding 的逻辑，但需要适配多维输入
        # 这里用一种简单的高频映射策略: [sin(2^k * PI * x), cos(2^k * PI * x), ...]
        
        # 简单起见，我们对每一列分别做 timestep_embedding 然后拼接
        # 假设 x[:, 0] 是实部， x[:, 1] 是虚部
        freqs = []
        for i in range(x.shape[1]):
            # 使用 TimestepEmbedder 的静态方法 (需要确保 import 了)
            emb = TimestepEmbedder.timestep_embedding(x[:, i], self.freq_embedding_size)
            freqs.append(emb)
        return torch.cat(freqs, dim=-1) # [N, 2 * freq_size]

    def forward(self, labels, train, force_drop_ids=None):
        # 1. 先进行高维映射 (Fourier Features)
        # labels: [N, 2] -> [N, 512]
        labels_freq = self.sinusoidal_embedding(labels)
        
        # 2. 再过 MLP
        embeddings = self.mlp(labels_freq)

        # 3. 处理 Dropout (和之前一样，替换输出)
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            if force_drop_ids is None:
                drop_ids = torch.rand(embeddings.shape[0], device=embeddings.device) < self.dropout_prob
            else:
                drop_ids = force_drop_ids == 1
            
            if drop_ids.any():
                embeddings = torch.where(
                    drop_ids.unsqueeze(1), 
                    self.null_embedding.to(embeddings.dtype), 
                    embeddings
                )
        return embeddings

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2 * hidden_size, 6 * hidden_size, bias=True)
            # nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # MSA 是 Multi-Head Self-Attention 的缩写，MLP 是 Multi-Layer Perceptron 的缩写
        # shift_msa 等六个参数其实是和 c 等长的向量，是 c 经过激活函数和线性变换后的结果
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2 * hidden_size, 2 * hidden_size, bias=True)
            # nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=8, # 我将输入图片尺寸从32改为8，因为声学超材料输入的图片是8x8的
        patch_size=1, # 我将patch_size改为1，因为声学超材料输入的图片是8x8的已经很小了
        in_channels=1, # 我将输入通道数从4改为1，因为声学超材料输入的图片是单通道的
        hidden_size=1152,
        depth=28, # 决定单个DiT类中堆叠了多少个DiTBlock模块
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        learnable_null=False,  # <--- [新增参数] 默认为 False，保证向后兼容
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        # # [修改] 根据参数选择使用哪个 Embedder
        # if learnable_null:
        #     print(" [Model] Using Learnable Null Embedding (New Scheme)")
        #     self.y_embedder = LearnableContinuousEmbedder(input_dim=2, hidden_size=hidden_size, dropout_prob=class_dropout_prob)
        # else:
        #     print(" [Model] Using Fixed Null Input [2, 2] (Legacy Scheme)")
        #     self.y_embedder = ContinuousEmbedder(input_dim=2, hidden_size=hidden_size, dropout_prob=class_dropout_prob)
        # [修正] 使用 FourierContinuousEmbedder
        if learnable_null:
            print(" [Model] Using Fourier Features + Learnable Null Embedding")
            # 注意：FourierContinuousEmbedder 内部已经包含了 learnable null 逻辑
            self.y_embedder = FourierContinuousEmbedder(input_dim=2, hidden_size=hidden_size, dropout_prob=class_dropout_prob)
        else:
            print(" [Model] Using Fixed Null Input [2, 2] (Legacy Scheme)")
            self.y_embedder = ContinuousEmbedder(input_dim=2, hidden_size=hidden_size, dropout_prob=class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # range(depth)：创建一个从 0 到 depth-1 的迭代器
        # _ 是一个占位符变量名，表示我们在循环中并不关心当前的索引值，只是单纯地想重复执行创建操作
        # nn.ModuleList 会将列表中所有的 DiTBlock 自动注册为模型的子模块
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # [修改] 针对不同 Embedder 的初始化逻辑
        if isinstance(self.y_embedder, LearnableContinuousEmbedder) or isinstance(self.y_embedder, FourierContinuousEmbedder):
            # 初始化 MLP
            for module in self.y_embedder.mlp:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            # [核心] 初始化可学习的 Null Embedding
            nn.init.normal_(self.y_embedder.null_embedding, std=0.02)
            
        else:
            # 旧逻辑 (ContinuousEmbedder)
            for module in self.y_embedder.mlp:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    # [修改] 增加 force_drop_ids 参数，用于接收强制丢弃信号
    def forward(self, x, t, y, force_drop_ids=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        force_drop_ids: (N,) tensor of booleans, True indicates this sample should use null embedding
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        
        # [核心修改] 将 force_drop_ids 传递给 y_embedder
        # 这样 LearnableContinuousEmbedder 才能知道何时使用 null_embedding
        y = self.y_embedder(y, self.training, force_drop_ids)    # (N, D)
        
        # c = t + y                                # (N, D)
        # [修改] 改为拼接
        # c 的形状变为 [N, 2 * hidden_size]
        c = torch.cat([t, y], dim=-1)

        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    # [修改] 在这里自动构造 force_drop_ids
    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        
        # [核心修改] 构造 force_drop_ids
        # combined 的前半部分是条件生成 (False)，后半部分是无条件生成 (True)
        force_drop_ids = torch.cat([
            torch.zeros(len(half), dtype=torch.bool, device=x.device),
            torch.ones(len(half), dtype=torch.bool, device=x.device)
        ], dim=0)

        # 将 force_drop_ids 传给 forward
        model_out = self.forward(combined, t, y, force_drop_ids=force_drop_ids)
        
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def DiT_Tiny1(**kwargs):
    # hidden_size=64, depth=6 左右即可，避免过拟合
    return DiT(depth=6, hidden_size=64, patch_size=1, num_heads=4, **kwargs)

DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
    'DiT-Tiny1': DiT_Tiny1,
}
