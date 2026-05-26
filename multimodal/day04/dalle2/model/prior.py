import torch
import torch.nn as nn
from model.clip import CLIP
from data.data_utils import *
from model.transformer import SinusoidalPositionalEmbedding, TransformerBlock

class DiffusionPrior(nn.Module):
    """先验模型结构"""
    def __init__(self, config):
        super().__init__()
        # 加载并冻结clip模型
        self.clip = CLIP(config).to(config.device)
        self.clip.load_state_dict(torch.load(config.clip.model_location, map_location=config.device))
        freeze_model(self.clip)

        self.config = config

        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(config.prior.max_time, config.latent_dim),
            nn.Linear(config.latent_dim, config.latent_dim * config.prior.r_mlp, bias=config.prior.bias),
            nn.SiLU(),
            nn.Linear(config.latent_dim * config.prior.r_mlp, config.latent_dim, bias=config.prior.bias)
        )
        # 可学习嵌入
        self.learned_embedding = nn.Parameter(torch.randn(config.latent_dim))

        self.schedule_values = get_schedule_values(schedule=config.prior.schedule, max_time=config.prior.max_time, device=config.device)

        # Transformer blocks
        self.decoder = nn.ModuleList(
            [TransformerBlock(
                config.latent_dim,
                cond_width=config.latent_dim,
                n_heads=config.prior.n_heads,
                dropout=config.prior.dropout,
                r_mlp=config.prior.r_mlp,
                bias=config.prior.bias
            ) for _ in range(config.prior.n_layers)]
        )

        # Output Projection
        self.output = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.latent_dim, bias=config.decoder.bias)
        )

        self.register_buffer("causal_attention_mask", torch.tril(torch.ones(5, 5))[None, :])

    def get_one_sample(self, text_embeddings, captions):
        """采样一个输出（预测的clip图片特征向量）的代码"""
        # 和clip图片特征向量形状一样的白噪声
        noisy_image_embeddings = torch.randn(text_embeddings.shape, device=self.config.device)

        # 最大时间步1000
        timesteps = torch.full((captions.shape[0],), self.config.prior.max_time - 1)

        # 获取时间步的嵌入
        timestep_embeddings = self.time_mlp(timesteps) # (B, ) -> (B, latent_dim)
        timestep_embeddings = timestep_embeddings[:, None, :] # (B, latent_dim) -> (B, 1, latent_dim)

        # 可学习嵌入
        learned_embeddings = self.learned_embedding.repeat(captions.shape[0], 1) # (latent_dim) -> (B, latent_dim)
        learned_embeddings = learned_embeddings[:, None, :] # (B, latent_dim) -> (B, 1, latent_dim)

        tokens = torch.cat((
            captions,               # 图片标题(256)
            text_embeddings,        # 图片标题的clip文本特征向量(256)
            timestep_embeddings,    # 时间步嵌入(256)
            noisy_image_embeddings,  # 纯噪声
            learned_embeddings      # 可学习嵌入
        ), dim=1) # (B, 5, latent_dim)

        # Pass through transformer blocks with causal attention mask
        for block in self.decoder:
            tokens = block(tokens, mask=self.causal_attention_mask)

        # Get learned embeddings and pass through output projection to get CLIP image embeddings
        # 先得到可学习嵌入（索引为-1），然后经过self.output输出预测到的clip图片特征向量
        pred_image_embeddings = self.output(tokens[:, -1, :])

        return pred_image_embeddings

    def sample(self, captions, masks=None):
        """在推理时使用"""
        # Get CLIP text embeddings
        t_emb = self.clip.text_encoder(captions, mask=masks) # (B, text_seq_length) -> (B, latent_dim)
        text_embeddings = t_emb[:, None, :] # (B, latent_dim) -> (B, 1, latent_dim)

        # Make caption length equal to latent dimension
        # 是文本标题的向量为256个元素
        if self.config.text_seq_length >= self.config.latent_dim:
            captions = captions[:, :self.config.latent_dim]  # (B, max_seq_len) -> (B, latent_dim)
        else:
            captions = nn.functional.pad(captions, (0, self.config.latent_dim - self.config.text_seq_length)) # (B, max_seq_len) -> (B, latent_dim)

        captions = captions[:, None, :] # (B, latent_dim) -> (B, 1, latent_dim)

        # Getting two samples
        sample_1 = self.get_one_sample(text_embeddings, captions)
        sample_2 = self.get_one_sample(text_embeddings, captions)

        gen_image_embeddings = torch.zeros(sample_1.shape)

        # Choosing the samples with the higher dot product with text embeddings
        for i in range(gen_image_embeddings.shape[0]):
            if sample_1[i] @ t_emb[i] >= sample_2[i] @ t_emb[i]:
                gen_image_embeddings[i] = sample_1[i]
            else:
                gen_image_embeddings[i] = sample_2[i]

        return gen_image_embeddings

    def forward(self, images, captions, masks=None):
        """训练时使用"""
        # 获取clip图片的特征向量
        image_embeddings = self.clip.image_encoder(images) # (B, C, H, W) -> (B, latent_dim)

        # Get CLIP text embeddings
        text_embeddings = self.clip.text_encoder(captions, mask=masks) # (B, text_seq_length) -> (B, latent_dim)
        text_embeddings = text_embeddings[:, None, :] # (B, latent_dim) -> (B, 1, latent_dim)

        # Make caption length equal to latent dimension
        if self.config.text_seq_length >= self.config.latent_dim:
            captions = captions[:, :self.config.latent_dim]  # (B, max_seq_len) -> (B, latent_dim)
        else:
            captions = nn.functional.pad(captions, (0, self.config.latent_dim - self.config.text_seq_length)) # (B, max_seq_len) -> (B, latent_dim)

        captions = captions[:, None, :] # (B, latent_dim) -> (B, 1, latent_dim)

        # Get random timesteps for forward diffusion
        timesteps = torch.randint(0, self.config.prior.max_time, (images.shape[0],)) # (B, )

        # Get timestep embeddings
        timestep_embeddings = self.time_mlp(timesteps) # (B, ) -> (B, latent_dim)
        timestep_embeddings = timestep_embeddings[:, None, :] # (B, latent_dim) -> (B, 1, latent_dim)

        # Perform forward diffusion to get noisy CLIP image embeddings
        noisy_image_embedding, _ = forward_diffusion(image_embeddings, self.schedule_values, timesteps)
        noisy_image_embedding = noisy_image_embedding[:, None, :] # (B, latent_dim) -> (B, 1, latent_dim)

        # Expand learned embedding so that there is one for each item in batch
        learned_embeddings = self.learned_embedding.repeat(images.shape[0], 1) # (latent_dim) -> (B, latent_dim)
        learned_embeddings = learned_embeddings[:, None, :] # (B, latent_dim) -> (B, 1, latent_dim)

        tokens = torch.cat((
            captions,               # Image Caption
            text_embeddings,        # CLIP Text Embedding
            timestep_embeddings,    # Timestep Embedding
            noisy_image_embedding,  # Noisy CLIP Image Embedding
            learned_embeddings      # Learned Embedding
        ), dim=1) # (B, 5, latent_dim)

        # Pass through transformer blocks with causal attention mask
        for block in self.decoder:
            tokens = block(tokens, mask=self.causal_attention_mask)

        # Get learned embeddings and pass through output projection to get CLIP image embeddings
        pred_image_embeddings = self.output(tokens[:, -1, :])

        loss = nn.functional.mse_loss(pred_image_embeddings, image_embeddings)

        return loss