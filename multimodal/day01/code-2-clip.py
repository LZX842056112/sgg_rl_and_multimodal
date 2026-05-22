# ============================================================
# 导入必要的库
# ============================================================
import torch                                      # PyTorch 深度学习框架，提供张量运算和自动求导
import torch.nn as nn                             # 神经网络模块，包含 LayerNorm、Linear、Embedding 等层
import torch.optim as optim                       # 优化器模块，包含 Adam、SGD 等
import torchvision.transforms as T                # torchvision 图像变换工具，用于数据预处理
from torch.utils.data import Dataset, DataLoader  # Dataset：自定义数据集基类；DataLoader：批量加载器
from datasets import load_dataset                 # HuggingFace datasets 库，用于加载/管理数据集
import matplotlib.pyplot as plt                   # 绘图库（本段代码未实际使用，可忽略）
import numpy as np                                # NumPy 数值计算库，此处主要用于 np.sin / np.cos


# ============================================================
# 位置嵌入模块 (Positional Embedding)
# 功能：生成固定的正弦/余弦位置编码，为 Transformer 提供序列位置信息
# 原理：Transformer 本身是置换不变的（打乱 token 顺序输出不变），
#       位置编码为每个 token 注入其所在位置的信息
# 公式（原始 Transformer 论文 "Attention Is All You Need"）：
#   PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))   偶数维度用 sin
#   PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))   奇数维度用 cos
# ============================================================
class PositionalEmbedding(nn.Module):
    def __init__(self, width, max_seq_length):
        """
        初始化位置编码矩阵。
        参数:
            width:           嵌入维度 d_model（每个 token 的特征向量长度）
            max_seq_length:  支持的最大序列长度（单位：token 数量）
        """
        super().__init__()  # 调用父类 nn.Module 的构造函数，注册模块

        # 创建一个形状为 (max_seq_length, width) 的全零张量作为位置编码的容器
        # 行索引 pos：序列中的位置（0 到 max_seq_length-1）
        # 列索引 i:   嵌入向量的维度（0 到 width-1）
        pe = torch.zeros(max_seq_length, width)

        # 逐位置逐维度计算正/余弦位置编码值
        for pos in range(max_seq_length):          # 遍历序列的每个位置
            for i in range(width):                 # 遍历嵌入向量的每个维度
                if i % 2 == 0:                     # 偶数维度（0, 2, 4, ...）
                    # PE(pos, 2i) = sin(pos / 10000^(2i / width))
                    # 10000^(2i/width) 是频率的倒数，随着 i 增大，频率降低
                    pe[pos][i] = np.sin(pos / (10000 ** (i / width)))
                else:                               # 奇数维度（1, 3, 5, ...）
                    # PE(pos, 2i+1) = cos(pos / 10000^(2i / width))
                    # 注意：公式中用 (i-1) 而非 i，是为了让同一对 sin/cos 使用相同频率
                    pe[pos][i] = np.cos(pos / (10000 ** ((i - 1) / width)))

        # register_buffer 将 pe 注册为"缓冲区"（非参数张量）
        # 特性：① 随 model.to(device) 自动迁移到 GPU
        #       ② 保存/加载模型时自动包含在 state_dict 中
        #       ③ 不参与梯度计算（requires_grad=False），位置编码不做反向传播优化
        # unsqueeze(0)：在第 0 维增加 batch 维度
        #   (max_seq_length, width) → (1, max_seq_length, width)
        #   便于后续 forward 时通过广播与输入相加
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        前向传播：将位置编码直接加到输入张量上。
        参数:
            x:  形状 (batch_size, seq_len, width) 的 token 嵌入
        返回:
            x:  形状不变，已叠加位置信息
        """
        # self.pe 形状为 (1, max_seq_length, width)，通过广播机制：
        #   batch 维度：1 → 自动扩展到 batch_size
        #   seq 维度：  max_seq_length（需要与 x 的 seq_len 匹配）
        # 这种"相加"而非"拼接"的方式，在计算效率上更高（不增加维度）
        x = x + self.pe
        return x


# ============================================================
# 单头注意力机制 (Single-Head Attention)
# 功能：计算输入序列的自注意力，让每个 token 能"看到"序列中的其他 token
# 公式：Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
# ============================================================
class AttentionHead(nn.Module):
    def __init__(self, width, head_size):
        """
        初始化单头注意力层。
        参数:
            width:     输入的特征维度 d_model（完整嵌入维度）
            head_size: 该注意力头的内部维度 d_k（通常为 width // n_heads）
        """
        super().__init__()
        self.head_size = head_size  # 保存头维度，用于后续的缩放因子计算

        # Q/K/V 的线性投影矩阵
        # 每个都是将 width 维的输入映射到 head_size 维的子空间
        # 本质上是 nn.Linear 执行: output = input @ W^T + b
        # 其中 W 形状为 (head_size, width)，b 形状为 (head_size,)
        self.query = nn.Linear(width, head_size)   # 查询投影矩阵 W_Q
        self.key   = nn.Linear(width, head_size)   # 键投影矩阵   W_K
        self.value = nn.Linear(width, head_size)   # 值投影矩阵   W_V

    def forward(self, x, mask=None):
        """
        前向传播：计算缩放点积注意力。
        参数:
            x:    形状 (batch_size, seq_len, width) 的输入序列
            mask: 形状 (batch_size, seq_len, seq_len) 或兼容形状的掩码
                  值为 0 的位置会被忽略（注意力权重置为 0）
        返回:
            attention: 形状 (batch_size, seq_len, head_size) 的注意力输出
        """
        # 步骤 1：对输入做线性投影，得到 Q/K/V
        # Q/K/V 形状均为 (batch_size, seq_len, head_size)
        Q = self.query(x)   # 查询：当前 token "想要关注什么信息"
        K = self.key(x)     # 键：  每个 token "能提供什么信息"
        V = self.value(x)   # 值：  每个 token "实际携带什么内容"

        # 步骤 2：计算 Q 和 K 的点积，得到注意力分数矩阵
        # Q @ K^T：矩阵乘法
        #   Q 形状: (batch_size, seq_len, head_size)
        #   K.transpose(-2, -1): 交换 K 的最后两个维度
        #      (batch_size, seq_len, head_size) → (batch_size, head_size, seq_len)
        #   结果形状: (batch_size, seq_len, seq_len)
        # 结果矩阵第 (i, j) 个元素：第 i 个 token 对第 j 个 token 的"原始关注度"
        attention = Q @ K.transpose(-2, -1)

        # 步骤 3：缩放注意力分数
        # 除以 sqrt(d_k) 的原因：
        #   当 d_k 很大时，Q 和 K 的内积方差会增大（约等于 d_k）
        #   方差大会导致 softmax 的输出趋向于 one-hot 分布（梯度极小，难以训练）
        #   除以 sqrt(d_k) 将方差控制回 1，保证 softmax 的梯度稳定
        attention = attention / (self.head_size ** 0.5)

        # 步骤 4：应用掩码（仅在文本编码器中使用）
        # masked_fill: 将 mask==0 的位置替换为 -inf
        #   经过 softmax 后，-inf 对应的位置权重 → 0（因为 exp(-inf) = 0）
        #   这样填充 token 就不会影响实际内容的表示
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))

        # 步骤 5：softmax 归一化
        # dim=-1：在最后一个维度（seq_len，即每个 token 对所有 token 的关注维度）上做 softmax
        # 结果：每一行的数值和为 1，表示第 i 个 token 对各 token 的注意力权重分布
        attention = torch.softmax(attention, dim=-1)

        # 步骤 6：用注意力权重对 V 做加权求和
        # attention @ V:
        #   attention 形状: (batch_size, seq_len, seq_len) → 注意力权重矩阵
        #   V 形状:         (batch_size, seq_len, head_size) → 每个 token 的 value
        #   结果形状:        (batch_size, seq_len, head_size)
        # 第 i 个 token 的输出 = Σ_j (attention[i][j] * V[j])
        # 即用注意力权重对序列中所有 token 的 value 做加权聚合
        attention = attention @ V
        return attention


# ============================================================
# 多头注意力机制 (Multi-Head Attention)
# 功能：并行运行多个注意力头，让模型在多个不同的"表示子空间"中同时关注信息
# 例如：头1 关注语法关系，头2 关注语义相似性，头3 关注长距离依赖
# ============================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, width, n_heads):
        """
        初始化多头注意力层。
        参数:
            width:   输入的特征维度 d_model（必须能被 n_heads 整除）
            n_heads: 注意力头的数量
        """
        super().__init__()
        # 计算每个头的维度 d_k = d_model // h
        # 例如：width=32, n_heads=8 → head_size=4
        self.head_size = width // n_heads

        # W_o：输出投影矩阵，将拼接后的多头输出映射回 width 维
        # 形状：(width, width)，即 d_model × d_model
        # 作用：让模型学习如何融合不同头的信息
        self.W_o = nn.Linear(width, width)

        # 使用 ModuleList 管理多个注意力头
        # ModuleList 是 PyTorch 的容器，确保内部的子模块被正确注册
        #   （能被 model.parameters() 发现，能随 model.to() 迁移设备）
        # 列表推导式创建 n_heads 个独立的 AttentionHead
        self.heads = nn.ModuleList([
            AttentionHead(width, self.head_size) for _ in range(n_heads)
        ])

    def forward(self, x, mask=None):
        """
        前向传播：拼接所有注意力头的输出。
        参数:
            x:    形状 (batch_size, seq_len, width)
            mask: 可选的注意力掩码
        返回:
            out:  形状 (batch_size, seq_len, width)
        """
        # [head(x, mask=mask) for head in self.heads]
        #   列表推导式：对每个头分别计算注意力，得到 n_heads 个张量
        #   每个张量形状: (batch_size, seq_len, head_size)
        #
        # torch.cat(..., dim=-1)
        #   在最后一个维度拼接所有头的输出
        #   (batch_size, seq_len, head_size) × n_heads
        #   → 拼接后: (batch_size, seq_len, head_size * n_heads)
        #   = (batch_size, seq_len, width)
        out = torch.cat([head(x, mask=mask) for head in self.heads], dim=-1)

        # W_o 将拼接结果映射回 width 维，融合多头信息
        # 即使输入输出维度相同（都是 width），这个线性变换仍然很重要：
        # 它让模型学习如何最优地组合各头的信息
        out = self.W_o(out)
        return out


# ============================================================
# Transformer 编码器层 (Transformer Encoder Layer)
# 功能：一个完整的 Transformer 编码器块，包含自注意力和前馈网络
# 架构：Pre-LN (Pre-Layer Normalization)
#   x → LayerNorm → MultiHeadAttention → +x (残差) → LayerNorm → MLP → +x (残差) → 输出
# 每个子层都有残差连接：out = LayerNorm(x) + Sublayer(x)
# ============================================================
class TransformerEncoder(nn.Module):
    def __init__(self, width, n_heads, r_mlp=4):
        """
        初始化 Transformer 编码器层。
        参数:
            width:   输入特征维度 d_model
            n_heads: 多头注意力头数
            r_mlp:   MLP 隐藏层的维度放大系数（默认 4，即 d_ff = 4 * d_model）
        """
        super().__init__()
        self.width = width      # 保存嵌入维度，供 MLP 构建时使用
        self.n_heads = n_heads  # 保存头数

        # 子层 1：层归一化 → 多头注意力
        # LayerNorm 在特征维度上做归一化，让每个样本的所有特征均值为 0、方差为 1
        # 与 BatchNorm 的区别：BatchNorm 在 batch 维度归一化，LayerNorm 在特征维度归一化
        # Transformer 中用 LN 是因为序列长度可变，BN 在变长序列上不稳定
        self.ln1 = nn.LayerNorm(width)                # 第一个 LayerNorm

        # 多头注意力：n_heads 个头，每个头维度为 width // n_heads
        self.mha = MultiHeadAttention(width, n_heads)

        # 子层 2：层归一化 → 前馈网络 MLP
        self.ln2 = nn.LayerNorm(width)                # 第二个 LayerNorm

        # MLP (Feed-Forward Network) 的结构：
        #   Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model)
        # d_ff = width * r_mlp (通常是 4 * d_model)
        # 为什么需要 MLP？注意力只做线性加权，MLP 引入非线性变换增强表达能力
        self.mlp = nn.Sequential(
            nn.Linear(self.width, self.width * r_mlp),  # 第1层：升维 d → 4d
            # GELU (Gaussian Error Linear Unit):
            #   GELU(x) = x * Φ(x)，Φ 是标准正态分布的累积分布函数
            #   相比 ReLU (max(0, x))：GELU 在 0 附近是平滑的，梯度更稳定
            #   现代 Transformer (BERT、GPT、ViT) 普遍使用 GELU
            nn.GELU(),
            nn.Linear(self.width * r_mlp, self.width)   # 第2层：降维 4d → d
        )

    def forward(self, x, mask=None):
        """
        前向传播：Pre-Norm 残差架构。
        参数:
            x:    形状 (batch_size, seq_len, width)
            mask: 可选的注意力掩码
        返回:
            x:    形状 (batch_size, seq_len, width)
        """
        # 残差连接 1：x + MHA(LayerNorm(x))
        #   Pre-Norm 的做法：先归一化，再做注意力，最后加残差
        #   相比 Post-Norm (MHA(x) + x 再加 LN)，Pre-Norm 训练更稳定
        #   残差连接的作用：让梯度直接流过加法节点，缓解深层网络的梯度消失
        x = x + self.mha(self.ln1(x), mask=mask)

        # 残差连接 2：x + MLP(LayerNorm(x))
        #   MLP 对每个 token 独立操作（位置间不交互）
        #   残差连接同样保证梯度平滑回传
        x = x + self.mlp(self.ln2(x))
        return x


# ============================================================
# 简易字符级分词器 (Tokenizer)
# 功能：将文本编码为整数序列，或将整数序列解码回文本
# 方式：使用 UTF-8 编码，每个字节的值 (0~255) 作为一个 token ID
# 特殊 token：
#   SOT (Start of Text) = chr(2)，序列开始标记
#   EOT (End of Text)   = chr(3)，序列结束标记
#   PAD (Padding)       = chr(0)，填充标记
# ============================================================
def tokenizer(text, encode=True, mask=None, max_seq_length=32):
    """
    字符级分词器：编码/解码文本。

    参数:
        text:           编码时是原始文本字符串；解码时是 token ID 张量
        encode:         True = 编码（文本→token IDs），False = 解码（token IDs→文本）
        mask:           编码时通常不传（函数内部生成）；解码时需传入有效 token 的位置掩码
        max_seq_length: 最大序列长度，超出截断、不足则填充

    返回:
        编码时: (token_tensor: IntTensor, mask_tensor: IntTensor)
                token_tensor 形状 (max_seq_length,)，每个元素是 0~255 的 token ID
                mask_tensor 形状 (max_seq_length,)，有效位置为1、填充位置为0
        解码时: (decoded_text: str, None)
    """
    if encode:
        # --- 编码流程 ---

        # 1. 添加特殊标记
        # chr(2) = SOT (Start of Text)，   chr(3) = EOT (End of Text)
        # CLIP 论文中使用 [SOS] 和 [EOS]，这里用 ASCII 控制字符实现
        out = chr(2) + text + chr(3)

        # 2. 用 PAD token 填充到固定长度 max_seq_length
        # chr(0) = PAD，循环创建 max_seq_length - len(out) 个填充字符
        # "".join(...) 将它们拼成一个字符串追加在末尾
        out = out + "".join([chr(0) for _ in range(max_seq_length - len(out))])

        # 3. 将字符串编码为整数序列
        # out.encode("utf-8")：将每个字符编码为 UTF-8 字节串
        #   - ASCII 字符（包括控制字符 0~127）的 UTF-8 编码与 ASCII 相同，1 字节
        #   - 所以 chr(0)~chr(127) 的编码值就是 0~127
        # list(...)：将字节串转为整数列表
        # torch.IntTensor(...)：转为 PyTorch 的 32 位整数张量
        # 最终 out 形状: (max_seq_length,)，每个元素是 0~255 的整数
        out = torch.IntTensor(list(out.encode("utf-8")))

        # 4. 构建注意力掩码 (attention mask)
        # 有效 token 位置为 1，PAD 填充位置为 0
        # nonzero() 返回非零元素的索引。由于只有 chr(0)(值为0) 可能为零，
        # 所以 nonzero 的数量就是实际有效字符数。
        # ⚠️ 注意：这种方式有缺陷——如果某个合法的 token ID 碰巧是 0
        #    (即 chr(0) 出现在文本中)，会被误判为填充。
        #    但在此 MNIST 场景下，"An image of X" 的 UTF-8 编码不含 0，所以安全。
        mask = torch.ones(len(out.nonzero()))  # 有效 token 数量个 1
        # 用 zeros 填充剩余位置，使 mask 长度 = max_seq_length
        mask = torch.cat((
            mask,
            torch.zeros(max_seq_length - len(mask))
        )).type(torch.IntTensor)

    else:
        # --- 解码流程 ---

        # text 形状: (max_seq_length,)，是 token ID 序列
        # mask.nonzero() 返回有效位置的索引
        # text[1 : len(mask.nonzero())-1]：
        #   从索引 1 开始（跳过 SOT），到最后一个有效位置前结束（跳过 EOT）
        # chr(x)：将每个整数 token ID 转回对应的字符
        out = [chr(x) for x in text[1:len(mask.nonzero()) - 1]]
        # 拼接为完整字符串
        out = "".join(out)
        mask = None  # 解码时不需要返回 mask

    return out, mask


# ============================================================
# 文本编码器 (Text Encoder)
# 功能：将文本 token 序列编码为固定维度的特征向量
# 流程：Embedding → 位置编码 → N×Transformer → 取 EOT → 投影 → L2归一化
# ============================================================
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, width, max_seq_length, n_heads, n_layers, emb_dim):
        """
        初始化文本编码器。
        参数:
            vocab_size:      词汇表大小（256，对应 UTF-8 字节范围）
            width:           嵌入维度 d_model
            max_seq_length:  最大序列长度
            n_heads:         注意力头数量
            n_layers:        Transformer 编码器层数
            emb_dim:         最终输出的多模态嵌入维度
        """
        super().__init__()
        self.max_seq_length = max_seq_length  # 保存最大序列长度

        # 词嵌入层 (Token Embedding)
        # 输入：token ID (0 ~ vocab_size-1 的整数)
        # 输出：width 维的稠密向量
        # 本质是一个查找表：shape (vocab_size, width)
        #   输入 token ID → 查出对应行的嵌入向量
        self.encoder_embedding = nn.Embedding(vocab_size, width)

        # 位置嵌入：在 token 嵌入上叠加位置信息
        self.positional_embedding = PositionalEmbedding(width, max_seq_length)

        # Transformer 编码器堆叠：n_layers 层
        # 使用 ModuleList 管理，确保各层参数被正确注册
        self.encoder = nn.ModuleList([
            TransformerEncoder(width, n_heads)
            for _ in range(n_layers)
        ])

        # 可学习的投影矩阵 (Projection Matrix)
        # 将文本特征从 d_model 维映射到 emb_dim 维（多模态嵌入空间）
        # 使用 nn.Parameter 而非 nn.Linear 的原因：
        #   nn.Parameter(torch.randn(width, emb_dim)) 只定义权重矩阵，没有 bias
        #   这是一种简化设计，实际 CLIP 论文中也使用类似的线性投影
        # 形状: (width, emb_dim)，即 (d_model, emb_dim)
        self.projection = nn.Parameter(torch.randn(width, emb_dim))

    def forward(self, text, mask=None):
        """
        前向传播：将文本编码为嵌入向量。
        参数:
            text: 形状 (batch_size, seq_len) 的 token ID 序列
            mask: 形状 (batch_size, seq_len) 的注意力掩码
        返回:
            x:    形状 (batch_size, emb_dim) 的文本嵌入向量（已 L2 归一化）
        """
        # 步骤 1：词嵌入
        # 将每个 token ID 替换为对应的 width 维稠密向量
        # (batch_size, seq_len) → (batch_size, seq_len, width)
        x = self.encoder_embedding(text)

        # 步骤 2：叠加位置编码
        # 将正弦位置编码加到词嵌入上，注入序列位置信息
        # 形状不变: (batch_size, seq_len, width)
        x = self.positional_embedding(x)

        # 步骤 3：通过多层 Transformer 编码器
        # 每层内部包含：多头自注意力 + MLP + 残差连接
        # 同时传递 mask 以保证 PAD token 不被其他 token 关注
        for encoder_layer in self.encoder:
            x = encoder_layer(x, mask=mask)

        # 步骤 4：提取 EOT token 的表示作为整个文本的特征
        # CLIP 论文中，句子的整体表示取 EOT (End of Text) token 对应的输出
        #
        # 索引计算：
        #   torch.sum(mask[:, 0], dim=1):
        #     mask[:, 0] 取掩码的第 0 列 → 形状 (batch_size,)
        #     .sum(dim=1) 对各行的第 0 列求和
        #     由于 mask 在 Dataset 中被 repeat 成了方阵，第 0 列标记了 token 0 与各 token
        #     的关系，但本质上有效位置的计数还是依赖于有效 token 的数量
        #     每个非填充 token 对应的 mask 为 1，总和 = 有效 token 数量
        #   torch.sub(..., 1):
        #     减 1 是因为要取最后一个有效 token（EOT），索引从 0 开始
        #     例如：有 5 个有效 token (SOT + "An" + "image" + "of" + "X" + EOT 可能)
        #          则 EOT 索引 = 5-1 = 4
        #   torch.arange(text.shape[0]):
        #     生成 [0, 1, 2, ..., batch_size-1]，用于批次中每个样本的行索引
        x = x[
            torch.arange(text.shape[0]),                      # batch 维索引: [0,1,...,B-1]
            torch.sub(torch.sum(mask[:, 0], dim=1), 1)        # 每个样本的 EOT 位置索引
        ]
        # 此时 x 形状: (batch_size, width)

        # 步骤 5：通过投影矩阵映射到多模态嵌入空间
        # x @ self.projection:
        #   (batch_size, width) @ (width, emb_dim) → (batch_size, emb_dim)
        # 投影到 emb_dim 维空间，与图像编码器输出维度一致，便于计算相似度
        if self.projection is not None:
            x = x @ self.projection

        # 步骤 6：L2 归一化
        # torch.norm(x, dim=-1, keepdim=True):
        #   计算每个向量的 L2 范数（欧几里得长度）
        #   dim=-1 在最后一个维度上计算
        #   keepdim=True 保持维度以便广播除法
        # 归一化后，所有嵌入向量落在单位超球面上
        #   ||x||_2 = 1
        #   此时两个向量的内积 = 余弦相似度（因为 ||a||=||b||=1）
        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x


# ============================================================
# 图像编码器 (Image Encoder) —— Vision Transformer (ViT) 风格
# 功能：将图像编码为固定维度的特征向量
# 流程：Patch分割(Conv2d) → Flatten → +CLS Token → 位置编码 → N×Transformer → 取CLS → 投影 → L2归一化
# ViT 论文："An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021)
# ============================================================
class ImageEncoder(nn.Module):
    def __init__(self, width, img_size, patch_size, n_channels, n_layers, n_heads, emb_dim):
        """
        初始化图像编码器（ViT 风格）。
        参数:
            width:       补丁嵌入的维度 d_model
            img_size:    输入图像尺寸 (H, W)
            patch_size:  每个补丁的尺寸 (Ph, Pw)，图像被划分为不重叠的补丁
            n_channels:  图像通道数（MNIST 灰度图为 1，RGB 图为 3）
            n_layers:    Transformer 编码器层数
            n_heads:     注意力头数量
            emb_dim:     最终输出的多模态嵌入维度
        """
        super().__init__()

        # 断言 1：图像尺寸必须能被补丁尺寸整除
        # 如果 img_size=(28,28), patch_size=(14,14)，则 28%14=0，通过
        # 这保证了图像可以被恰好划分为整数个不重叠的补丁
        assert img_size[0] % patch_size[0] == 0 \
            and img_size[1] % patch_size[1] == 0, \
            "img_size必须能被patch_size整除"

        # 断言 2：嵌入维度必须能被注意力头数整除
        # 因为多头注意力要求 width // n_heads 能均匀分配
        # 例如：width=9, n_heads=3 → head_size=3，通过
        assert width % n_heads == 0, \
            "width必须能被n_heads整除"

        # 计算补丁数量
        # 图像总面积 / 每个补丁的面积 = 补丁个数
        # 对于 MNIST 28×28, patch 14×14: n_patches = (28*28)/(14*14) = 784/196 = 4
        self.n_patches = (img_size[0] * img_size[1]) \
                       // (patch_size[0] * patch_size[1])

        # 序列最大长度 = 1 (CLS token) + 补丁数量
        # 对于 MNIST: max_seq_length = 1 + 4 = 5
        self.max_seq_length = self.n_patches + 1

        # 补丁嵌入层 (Patch Embedding)
        # 使用 Conv2d 实现无重叠的图像分块和线性投影
        #   nn.Conv2d(n_channels, width, kernel_size=patch_size, stride=patch_size)
        #
        #   为什么用卷积？因为卷积天然实现了"滑动窗口提取+线性投影"：
        #     kernel_size=patch_size → 每个卷积窗口就是一个补丁
        #     stride=patch_size      → 窗口之间不重叠（步长等于窗口大小）
        #     n_channels → width    → 每个补丁被投影到 width 维向量
        #
        #   对于 MNIST：
        #     输入: (batch, 1, 28, 28)
        #     卷积: Conv2d(1, 9, kernel=14, stride=14)
        #     输出: (batch, 9, 2, 2)   ← 9 个通道，2×2 的空间位置（4 个补丁）
        self.linear_project = nn.Conv2d(
            n_channels, width,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 分类 token (CLS Token)
        # 借鉴 BERT 的 [CLS] token 思想：在序列最前面添加一个可学习的 token
        # 经过 Transformer 编码后，CLS token 的输出汇聚了全局图像信息
        # 初始化为随机值 (1, 1, width)，后续通过训练学习最优表示
        #   nn.Parameter 确保这个张量参与梯度更新
        self.cls_token = nn.Parameter(torch.randn(1, 1, width))

        # 位置嵌入：为包括 CLS token 在内的所有 token 添加位置信息
        self.positional_embedding = PositionalEmbedding(
            width, self.max_seq_length
        )

        # Transformer 编码器堆叠
        # 注意：图像编码器不需要掩码（所有补丁都是有效的）
        self.encoder = nn.ModuleList([
            TransformerEncoder(width, n_heads)
            for _ in range(n_layers)
        ])

        # 可学习投影矩阵：将图像特征映射到多模态嵌入空间
        # 形状: (width, emb_dim)，即 (d_model, emb_dim)
        self.projection = nn.Parameter(torch.randn(width, emb_dim))

    def forward(self, x):
        """
        前向传播：将图像编码为嵌入向量。
        参数:
            x: 形状 (batch_size, n_channels, H, W) 的图像张量
        返回:
            x: 形状 (batch_size, emb_dim) 的图像嵌入向量（已 L2 归一化）
        """
        # 步骤 1：补丁嵌入 (Patch Embedding)
        # 卷积将图像分割为补丁并做线性投影
        #   (batch, C, H, W) → (batch, width, H/patch_h, W/patch_w)
        #   MNIST: (batch, 1, 28, 28) → (batch, 9, 2, 2)
        x = self.linear_project(x)

        # 步骤 2：展平空间维度并转置
        # flatten(2): 从第 2 维（索引从 0 开始）开始展平
        #   (batch, width, 2, 2) → (batch, width, 4)
        # transpose(1, 2): 交换第 1 和第 2 维
        #   (batch, width, 4) → (batch, 4, width)
        # 此时每个补丁是一个 width 维的 token，共 4 个补丁 token
        x = x.flatten(2).transpose(1, 2)

        # 步骤 3：在序列开头拼接 CLS token
        # self.cls_token 形状: (1, 1, width)
        # .expand(x.size(0), -1, -1):
        #   将 batch 维度从 1 扩展到 batch_size
        #   -1 表示保持该维度不变
        #   结果形状: (batch_size, 1, width)
        # torch.cat(..., dim=1)：在序列维度（dim=1）上拼接
        #   (batch, 1, width) + (batch, 4, width) → (batch, 5, width)
        #   序列变为: [CLS, patch_0, patch_1, patch_2, patch_3]
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)

        # 步骤 4：添加位置嵌入
        # 为 CLS token（位置 0）和 4 个补丁 token（位置 1~4）添加位置信息
        x = self.positional_embedding(x)

        # 步骤 5：通过多层 Transformer 编码器
        # 不需要 mask，因为图像所有补丁都是"有效"的，没有填充
        for encoder_layer in self.encoder:
            x = encoder_layer(x)

        # 步骤 6：取出 CLS token 的输出作为整张图像的特征
        # x[:, 0, :]:
        #   第 0 维（batch）：取所有样本
        #   第 1 维（seq）  ：取第 0 个位置（CLS token）
        #   第 2 维（特征） ：取所有特征维度
        # 结果形状: (batch_size, width)
        x = x[:, 0, :]

        # 步骤 7：投影到多模态嵌入空间
        # (batch_size, width) @ (width, emb_dim) → (batch_size, emb_dim)
        if self.projection is not None:
            x = x @ self.projection

        # 步骤 8：L2 归一化，将嵌入映射到单位超球面
        # 归一化后内积 = 余弦相似度
        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x


# ============================================================
# CLIP 模型 (Contrastive Language-Image Pre-training)
# 功能：联合训练图像和文本编码器，通过对比学习学习多模态对齐
# 核心思想：让匹配的 (图像, 文本) 对的嵌入向量相似度最大，
#          让不匹配的 (图像, 文本) 对的相似度最小
# 论文："Learning Transferable Visual Models From Natural Language
#        Supervision" (Radford et al., 2021)
# ============================================================
class CLIP(nn.Module):
    def __init__(self, emb_dim, vit_width, img_size, patch_size, n_channels,
                 vit_layers, vit_heads, vocab_size, text_width, max_seq_length,
                 text_heads, text_layers):
        """
        初始化 CLIP 模型。
        参数:
            emb_dim:         多模态嵌入空间的维度（图像/文本编码器输出维度相同）
            vit_width:       图像编码器的宽度 d_model
            img_size:        图像尺寸 (H, W)
            patch_size:      图像补丁尺寸 (Ph, Pw)
            n_channels:      图像通道数
            vit_layers:      图像编码器层数
            vit_heads:       图像编码器注意力头数
            vocab_size:      文本词汇表大小
            text_width:      文本编码器的宽度 d_model
            max_seq_length:  文本最大序列长度
            text_heads:      文本编码器注意力头数
            text_layers:     文本编码器层数
        """
        super().__init__()

        # 图像编码器（ViT 风格）
        # 将图像编码为 emb_dim 维的嵌入向量
        self.image_encoder = ImageEncoder(
            vit_width, img_size, patch_size, n_channels,
            vit_layers, vit_heads, emb_dim
        )

        # 文本编码器（Transformer 风格）
        # 将文本编码为 emb_dim 维的嵌入向量（与图像编码器输出维度相同）
        self.text_encoder = TextEncoder(
            vocab_size, text_width, max_seq_length,
            text_heads, text_layers, emb_dim
        )

        # 可学习的温度参数 τ (temperature)
        # 初始值: log(1/0.07) ≈ log(14.286) ≈ 2.659
        # 实际温度值: exp(log(1/0.07)) = 1/0.07 ≈ 14.286
        #
        # 温度的作用：
        #   logits = (I_e · T_e) / τ，其中 τ=exp(temperature)
        #   温度越小（τ 越小），相似度被放大越多，softmax 分布越"尖锐"
        #   这意味着模型对负样本的惩罚更严厉
        #
        # CLIP 论文中初始温度约为 0.07（即 logits 被放大约 14 倍）
        # 这里用对数参数化，让训练更稳定 (exp 确保温度始终为正)
        # torch.ones([]) 创建一个标量张量 (0 维)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 设备：硬编码为 cuda
        # ⚠️ 注意：更好的做法是用 torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda")

    def forward(self, image, text, mask=None):
        """
        前向传播：计算对比损失 (contrastive loss)。
        参数:
            image: 形状 (batch_size, C, H, W) 的图像批次
            text:  形状 (batch_size, seq_len) 的文本 token ID 序列
            mask:  形状 (batch_size, seq_len, seq_len) 的注意力掩码
        返回:
            loss:  标量张量，对比损失值
        """
        # 步骤 1：提取图像嵌入 I_e
        # 经过 ImageEncoder: 补丁嵌入 → Transformer → CLS → 投影 → L2归一化
        # 形状: (batch_size, emb_dim)
        I_e = self.image_encoder(image)

        # 步骤 2：提取文本嵌入 T_e
        # 经过 TextEncoder: 词嵌入 → 位置编码 → Transformer → EOT → 投影 → L2归一化
        # 形状: (batch_size, emb_dim)
        T_e = self.text_encoder(text, mask=mask)

        # 步骤 3：计算缩放后的余弦相似度矩阵
        # I_e @ T_e.transpose(-2, -1):
        #   I_e 形状:             (batch_size, emb_dim)
        #   T_e.transpose(-2,-1): (emb_dim, batch_size)
        #   结果: (batch_size, batch_size)，即 B×B 的相似度矩阵
        #   矩阵元素 logits[i][j] = I_e[i] · T_e[j]（向量内积 = 余弦相似度，因为已归一化）
        #
        # * torch.exp(self.temperature):
        #   exp(log(1/0.07)) = 1/0.07 ≈ 14.286
        #   将余弦相似度（范围 [-1, 1]）缩放为约 [-14, 14]
        #   缩放后的值送入 softmax，让分布更"尖锐"
        logits = (I_e @ T_e.transpose(-2, -1)) * torch.exp(self.temperature)

        # 步骤 4：构造标签
        # labels = [0, 1, 2, ..., B-1] 表示对角线位置
        # 在 B×B 的相似度矩阵中：
        #   logits[0][0] 是图像 0 和文本 0 的相似度（正样本对）
        #   logits[0][j] (j≠0) 是图像 0 和文本 j 的相似度（负样本对）
        # 标签 i 表示"第 i 个文本是正确的匹配"
        labels = torch.arange(logits.shape[0]).to(self.device)

        # 步骤 5：对称对比损失 (Symmetric Cross-Entropy Loss)
        #
        # 方向 1 —— 图像 → 文本 (Image-to-Text):
        #   logits.transpose(-2, -1) 将相似度矩阵转置
        #   此时第 i 行 = 图像 i 对所有文本的相似度
        #   交叉熵的目标：让图像 i 与文本 i（对角线）相似度最大
        #   即鼓励 I₀↔T₀, I₁↔T₁, ..., I_{B-1}↔T_{B-1} 匹配
        loss_i = nn.functional.cross_entropy(logits.transpose(-2, -1), labels)

        # 方向 2 —— 文本 → 图像 (Text-to-Image):
        #   logits 第 i 行 = 文本 i 对所有图像的相似度
        #   交叉熵的目标：让文本 i 与图像 i（对角线）相似度最大
        #   与方向 1 对偶但不等价（因为 softmax 在不同维度做）
        loss_t = nn.functional.cross_entropy(logits, labels)

        # 取两个方向的平均作为最终损失
        # 对称设计确保图像和文本之间的对齐是双向的
        loss = (loss_i + loss_t) / 2

        return loss


# ============================================================
# 自定义 MNIST 数据集类
# 功能：加载 MNIST 数据集，为每张图片生成对应的文本描述
# ============================================================
class MNIST(Dataset):
    def __init__(self, train=True):
        """
        初始化 MNIST 数据集。
        参数:
            train: True 加载训练集，False 加载测试集
        """
        # 使用 HuggingFace datasets 库从本地路径加载数据集
        # "./clip-mnist/" 是本地数据集目录路径
        self.dataset = load_dataset("./clip-mnist/")

        # ToTensor() 转换器：
        #   将 PIL Image 或 numpy 数组 (H, W, C) 转为 PyTorch Tensor (C, H, W)
        #   同时将像素值从 [0, 255] 缩放为 [0.0, 1.0] 浮点数
        self.transform = T.ToTensor()

        # 根据 train 参数选择 "train" 或 "test" 分割
        if train:
            self.split = "train"
        else:
            self.split = "test"

        # 定义 10 个数字对应的文本描述（英语描述格式，类似 CLIP 的 prompt 模板）
        # 用于对比学习：模型需要学会将"An image of 5"和手写数字 5 的图片对齐
        self.captions = {
            0: "An image of 0",
            1: "An image of 1",
            2: "An image of 2",
            3: "An image of 3",
            4: "An image of 4",
            5: "An image of 5",
            6: "An image of 6",
            7: "An image of 7",
            8: "An image of 8",
            9: "An image of 9"
        }

    def __len__(self):
        # 返回数据集的样本数量
        # .num_rows[self.split] 获取对应分割的样本数
        return self.dataset.num_rows[self.split]

    def __getitem__(self, i):
        """
        获取第 i 个样本。
        参数:
            i: 样本索引
        返回:
            字典 {"image": Tensor, "caption": Tensor, "mask": Tensor}
        """
        # 1. 获取第 i 张图像（PIL Image 格式）
        img = self.dataset[self.split][i]["image"]

        # 2. 将图像转为 Tensor，形状 (1, 28, 28)，值域 [0, 1]
        img = self.transform(img)

        # 3. 根据标签获取对应的文本描述并编码
        # self.captions[label] 获取文本，如 "An image of 5"
        # tokenizer(...) 返回 (token_sequence, mask)
        caption_text = self.captions[self.dataset[self.split][i]["label"]]
        cap, mask = tokenizer(caption_text)

        # 4. 将掩码 repeat 成方阵 (seq_len, seq_len)
        # mask 原始形状: (max_seq_length,)  例如 (32,)
        # mask.repeat(len(mask), 1):
        #   repeat 的第一个参数沿 dim=0 重复 len(mask) 次
        #   第二个参数 1 沿 dim=1 保持
        #   结果形状: (max_seq_length, max_seq_length)  即 (32, 32)
        #
        # 为什么需要方阵？
        #   在 AttentionHead.forward 中：
        #     attention @ K^T 形状为 (B, seq_len, seq_len)
        #     mask 需要能广播到这个形状，用于 masked_fill
        #   方阵 mask 的第 (i, j) 位置为 1 表示 token i 可以关注 token j
        #   第 (i, j) 位置为 0 表示 token i 不能关注 token j（PAD 位置）
        mask = mask.repeat(len(mask), 1)

        # 返回一个包含图像、文本和掩码的字典
        return {"image": img, "caption": cap, "mask": mask}


# ============================================================
# 超参数配置
# ============================================================
emb_dim = 32              # 多模态嵌入维度：图像和文本编码后都投影到这个维度
vit_width = 9             # 图像编码器宽度 d_model（每个补丁 token 的嵌入维度）
img_size = (28, 28)       # 输入图像尺寸（MNIST 为 28×28 灰度图）
patch_size = (14, 14)     # 补丁尺寸，28/14=2，产生 2×2=4 个补丁
n_channels = 1            # 图像通道数，MNIST 为灰度图（单通道）
vit_layers = 3            # 图像编码器的 Transformer 层数
vit_heads = 3             # 图像编码器的注意力头数（9/3=3，每个头维度=3）
vocab_size = 256          # 文本词汇表大小（UTF-8 字节范围 0~255）
text_width = 32           # 文本编码器宽度 d_model（每个 token 的嵌入维度）
max_seq_length = 32       # 文本最大序列长度（"An image of X" 约有 14 字节，32 足够）
text_heads = 8            # 文本编码器注意力头数（32/8=4，每个头维度=4）
text_layers = 4           # 文本编码器的 Transformer 层数
lr = 1e-3                 # 学习率 (learning rate)，Adam 优化器的初始学习率
epochs = 10               # 训练轮数（遍历整个数据集的次数）
batch_size = 128          # 批次大小（每次前向传播的样本数）

# ============================================================
# 数据准备
# ============================================================

# 创建训练集和测试集实例
train_set = MNIST(train=True)    # 训练集，用于模型参数优化
test_set = MNIST(train=False)    # 测试集，用于评估模型泛化能力

# DataLoader: 批量加载器，提供自动批处理、打乱、多线程加载等功能
#   shuffle=True:  训练时打乱数据顺序，防止模型记住数据顺序
#   shuffle=False: 测试时保持顺序，确保评估结果可复现
#   batch_size:    每次返回的样本数量
train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
test_loader  = DataLoader(test_set, shuffle=False, batch_size=batch_size)

# 设备：选择 GPU 进行训练
# ⚠️ 如果需要兼容无 GPU 环境，可使用:
#   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

# ============================================================
# 模型初始化
# ============================================================

# 实例化 CLIP 模型并移动到 GPU
# 模型参数通过之前定义的超参数传入
model = CLIP(
    emb_dim,         # 多模态嵌入维度
    vit_width,       # 图像编码器宽度
    img_size,        # 图像尺寸
    patch_size,      # 补丁尺寸
    n_channels,      # 图像通道数
    vit_layers,      # 图像编码器层数
    vit_heads,       # 图像编码器头数
    vocab_size,      # 词汇表大小
    text_width,      # 文本编码器宽度
    max_seq_length,  # 文本最大长度
    text_heads,      # 文本编码器头数
    text_layers      # 文本编码器层数
).to(device)         # 将模型的所有参数和 buffer 迁移到 GPU

# 优化器：Adam (Adaptive Moment Estimation)
# Adam 结合了 Momentum（动量）和 RMSprop（自适应学习率）的优点
# lr=1e-3 是 Adam 的推荐默认学习率
# model.parameters() 返回模型所有可训练参数的迭代器
optimizer = optim.Adam(model.parameters(), lr=lr)

# ============================================================
# 训练循环
# ============================================================

best_loss = np.inf   # 记录最佳损失值，初始化为正无穷（确保第一次一定会保存）

for epoch in range(epochs):                       # 外层循环：遍历每个 epoch
    for i, data in enumerate(train_loader, 0):    # 内层循环：遍历每个 batch
        # --- 数据准备：将数据移动到 GPU ---
        img = data["image"].to(device)    # 图像张量:     (batch, 1, 28, 28)
        cap = data["caption"].to(device)  # 文本 token ID: (batch, max_seq_length)
        mask = data["mask"].to(device)    # 注意力掩码:    (batch, max_seq_length, max_seq_length)

        # --- 前向传播：计算对比损失 ---
        # model(img, cap, mask) 内部执行：
        #   1. ImageEncoder → I_e
        #   2. TextEncoder → T_e
        #   3. 计算 I_e @ T_e^T 相似度矩阵
        #   4. 对称交叉熵损失
        loss = model(img, cap, mask)

        # --- 反向传播和参数更新 ---
        optimizer.zero_grad()   # 清空所有参数的梯度缓存（PyTorch 默认梯度是累加的）
        loss.backward()         # 反向传播：计算 loss 对所有模型参数的梯度
        optimizer.step()        # 参数更新：根据梯度使用 Adam 算法更新参数

    # --- 打印训练进度 ---
    # 只打印每个 epoch 最后一个 batch 的损失（简化输出）
    print(f"Epoch [{epoch+1}/{epochs}], Batch Loss: {loss.item():.3f}")

    # --- 模型保存：只保留损失最小的模型 ---
    if loss.item() <= best_loss:
        best_loss = loss.item()                           # 更新最佳损失
        torch.save(model.state_dict(), "./clip.pt")       # 保存模型权重到文件
        # state_dict() 是一个 Python 字典，将每一层映射到其参数张量
        # 保存为 .pt/.pth 文件，后续可通过 load_state_dict() 恢复
        print("模型已经保存.")

# ============================================================
# 模型评估 —— 零样本分类 (Zero-shot Classification)
# 原理：给定一张图像，计算它与所有 10 个类别文本描述的相似度，
#       选择相似度最高的类别作为预测结果
# ============================================================

# 重新创建模型实例并加载训练中保存的最佳权重
model = CLIP(
    emb_dim, vit_width, img_size, patch_size, n_channels,
    vit_layers, vit_heads, vocab_size, text_width, max_seq_length,
    text_heads, text_layers
).to(device)
# load_state_dict: 将保存的参数字典加载到模型中
# map_location=device: 确保权重加载到正确的设备上
model.load_state_dict(torch.load("./clip.pt", map_location=device))

# 预先编码所有 10 个类别的文本描述并转为 token 序列
# test_set.captions.values() 返回 10 个文本的描述列表
# [tokenizer(x)[0] for x in ...] 对每个文本编码，取 token ID（tokenizer 返回值的第 0 个元素）
# torch.stack(...) 将列表堆叠为张量
# text 形状: (10, max_seq_length)，即 10 个类别 × 32 个 token
text = torch.stack(
    [tokenizer(x)[0] for x in test_set.captions.values()]
).to(device)

# 获取文本对应的注意力掩码
# mask 形状: (10, max_seq_length)
mask = torch.stack(
    [tokenizer(x)[1] for x in test_set.captions.values()]
)
# 将掩码 repeat 成方阵，保持与训练时一致的形状
# repeat(1, len(mask[0])):
#   dim=0 保持 (1 → 不重复)
#   dim=1 重复 len(mask[0])=32 次
#   形状: (10, 32) → (10, 32*32) = (10, 1024)
# reshape(len(mask), len(mask[0]), len(mask[0])):
#   重新排列为 (10, 32, 32)，即每个类别的注意力掩码方阵
mask = mask.repeat(
    1, len(mask[0])
).reshape(
    len(mask),           # 10 个类别
    len(mask[0]),        # seq_len = 32
    len(mask[0])         # seq_len = 32
).to(device)

correct, total = 0, 0   # 正确预测数和总样本数
with torch.no_grad():   # 禁用梯度计算，节省显存并加速推理
    for data in test_loader:
        # 获取当前批次的数据
        images = data["image"].to(device)     # 图像张量: (batch, 1, 28, 28)
        labels = data["caption"].to(device)   # 真实文本 token 序列: (batch, max_seq_length)

        # 编码图像：每张图像获得一个 emb_dim 维的特征向量
        # 形状: (batch_size, emb_dim)
        image_features = model.image_encoder(images)

        # 编码文本：10 个类别的文本描述各自获得一个 emb_dim 维的特征向量
        # 形状: (10, emb_dim)
        text_features = model.text_encoder(text, mask=mask)

        # 再次 L2 归一化（编码器内部已经做过，这里是双重保险）
        # 确保范数严格为 1，避免数值误差导致余弦相似度计算不准确
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 计算图像与所有文本的余弦相似度
        # image_features @ text_features.T:
        #   (batch_size, emb_dim) @ (emb_dim, 10) → (batch_size, 10)
        # 第 i 行第 j 列 = 第 i 张图像与第 j 个类别文本的余弦相似度
        #
        # * 100.0：将相似度放大 100 倍
        #   余弦相似度范围 [-1, 1]，放大后为 [-100, 100]
        #   放大使 softmax 分布更"锐利"，更容易区分相似和不相似的类别
        #
        # .softmax(dim=-1)：在 10 个类别维度上做 softmax，转为概率分布
        #   每行的 10 个值之和为 1，最大值的类别即为预测结果
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # torch.max(similarity, 1):
        #   在第 1 维（类别维度）取最大值
        #   返回值: (values, indices)
        #     values:  每张图像的最大相似度值（忽略，用 _ 接收）
        #     indices: 最大值的索引（0~9），即预测的数字类别
        _, indices = torch.max(similarity, 1)

        # 将预测的类别索引转为对应的文本 token 序列
        # test_set.captions[int(i)] 根据预测的数字获取文本描述
        # tokenizer(...)[0] 编码为 token ID 序列
        # torch.stack(...) 堆叠为 (batch_size, max_seq_length)
        pred = torch.stack([
            tokenizer(test_set.captions[int(i)])[0]
            for i in indices
        ]).to(device)

        # 统计预测正确的样本数
        # pred == labels：逐元素比较，得到布尔张量 (batch_size, max_seq_length)
        # torch.sum(..., dim=1)：对每行（每个样本）求 True 的数量
        # // len(pred[0])：整除 max_seq_length
        #   如果所有 token 都匹配，结果为 1（完全正确）
        #   如果部分匹配，结果为 0（因为 token 数量 < max_seq_length 时不可能整除）
        # int(sum(...))：转为 Python 整数
        correct += int(sum(torch.sum((pred == labels), dim=1) // len(pred[0])))

        # 累加总样本数
        total += len(labels)

# 打印最终预测准确率（整数百分比）
# // 整数除法，结果显示为整数百分比（如 85 %）
print(f'\n预测准确率: {100 * correct // total} %')