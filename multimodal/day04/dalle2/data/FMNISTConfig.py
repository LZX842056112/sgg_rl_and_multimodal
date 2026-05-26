import torch
from dataclasses import dataclass, field

@dataclass
class CLIPConfig:
    # Vision Transformer
    # CLIP图片编码器的相关配置
    patch_size:tuple[int,int] = (4,4) # 补丁大小4x4
    vit_width:int = 256 # 图片补丁嵌入的维度
    vit_layers:int = 6
    vit_heads:int = 8
    # Text Transformer
    # CLIP文本编码器
    text_width:int = 256 # 词嵌入的维度
    text_layers:int = 6
    text_heads:int = 8
    # Attention
    dropout:float = 0.2
    r_mlp:int = 4
    bias:bool = False
    # Training
    augment_data:bool = True
    validate:bool = True
    num_workers:int = 0
    batch_size:int = 128
    lr:float = 5e-4
    lr_min:float = 1e-5
    weight_decay:float = 1e-4
    epochs:int = 200
    warmup_epochs:int = 5
    grad_max_norm:float = 1.0
    get_val_accuracy:bool = False
    model_location:str = "./trained_models/clip_fmnist.pt"

@dataclass
class PriorConfig:
    """先验模型的配置"""
    # Diffusion
    max_time:int = 1000 # 给clip图片的特征向量加噪声的最大时间步
    schedule:str = "cosine" # 使用余弦方差调度计划
    schedule_offset:float = 0.008
    # Transformer Decoder 仅解码器架构的transformer
    width:int = 256 # 先验模型输入的5个token，每个token的维度
    n_layers:int = 6
    n_heads:int = 8
    # Attention
    dropout:float = 0.2
    r_mlp:int = 4
    bias:bool = False
    # Training
    augment_data:bool = False
    validate:bool = True
    num_workers:int = 0
    batch_size:int = 128
    lr:float = 5e-4
    lr_min:float = 1e-5
    weight_decay:float = 1e-4
    epochs:int = 150
    warmup_epochs:int = 5
    grad_max_norm:float = 1.0
    model_location:str = "./trained_models/prior_fmnist.pt"

@dataclass
class DecoderConfig:
    """条件扩散模型"""
    # Diffusion
    max_time:int = 1000 # 给图片加噪声的最大时间步
    schedule:str = "cosine"
    # UNet
    n_groups:int = 8 # 组归一化中一组有8条数据
    kernel_size:tuple[int, int] = (3,3)
    model_channels:int = 32
    cond_channels:int = 128
    channel_ratios:list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    n_layer_blocks:int = 2
    dropout:float = 0.1
    use_scale_shift:bool = True
    n_heads:int = 8
    stride:int = 2
    down_pool:bool = False
    r_mlp:int = 4
    bias:bool = False
    text_layers:int = 4
    n_img_tokens:int = 4 # 图片特征向量要作为条件注入给注意力模块，需要转换成4个token
    # Training
    augment_data:bool = False
    validate:bool = False
    num_workers:int = 0
    batch_size:int = 32
    lr:float = 5e-4
    lr_min:float = 1e-5
    weight_decay:float = 1e-4
    epochs:int = 100
    warmup_epochs:int = 5
    grad_max_norm:float = 1.0
    sample_after_epoch:bool = False
    model_location:str = "./trained_models/decoder_fmnist.pt"

@dataclass
class FMNISTConfig:
    latent_dim:int = 256
    # Dataset Info
    dataset:str = "fashion_mnist"
    data_location:str = "./datasets"
    img_size:tuple[int,int] = (32,32)
    img_channels:int = 1
    vocab_size:int = 256 # 我们使用ascii码进行分词，所以词汇表大小256
    text_seq_length:int = 64 # 提示词最大序列长度为64
    # Data Augmentation / Normalization
    # 对图片进行数据增强/归一化的配置
    prob_hflip:float = 0.5
    crop_padding:int = 4
    train_mean:list[float] = field(default_factory=lambda: [0.2855552])
    train_std:list[float] = field(default_factory=lambda: [0.33848408])
    # Training
    train_val_split:tuple[int,int] = (50000, 10000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Model Configs
    clip = CLIPConfig()
    prior = PriorConfig()
    decoder = DecoderConfig()
