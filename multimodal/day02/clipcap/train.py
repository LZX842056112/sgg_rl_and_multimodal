import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from clipcap_dataset import ClipCapDataset
from model import ClipCaptionModel
import torch.nn.functional as F
from config import LLM_PATH, IMAGE_TOKEN_LENGTH, device


def train(model, train_loader, optimizer):
    model.train()
    for _ in range(20):
        for _, data in enumerate(tqdm(train_loader)):
            image_embed, caption_ids, mask = data
            image_embed = image_embed.to(device)
            caption_ids = caption_ids.to(device)
            mask = mask.to(device)
            # 输出的logits
            logits = model(image_embed, caption_ids, mask)

            # 计算loss
            # [图片的最后一个token]，[两]，[只]，[狗]
            #          ↓            ↓    ↓
            #         [两]         [只]  [狗]
            shift_logits = logits[
                :,
                # 截取范围[图片的最后一个token～倒数第二个token]
                IMAGE_TOKEN_LENGTH - 1: -1,  # 去掉最后一个token
                :
            ].contiguous().view(-1, logits.size(-1))
            # 预测目标
            shift_labels = caption_ids.view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels)
            # logits.size(-1): 取 logits 的最后一维大小。一般最后一维是词表大小 vocab_size。
            # 原 logits 形状通常是 [B, L, V]（批大小、序列长度、词表大小）。
            # 经过切片后，shift_logits 的形状是 [B, L-1, V]。
            # 再 `.contiguous().view(-1, logits.size(-1))`
            # 就变成 [B*(L-1), V]，把前两维展平，便于和标签做交叉熵。
            # caption_ids 形状通常是 [B, L-1]（与 shift_logits 的时间步对齐）。
            # caption_ids.view(-1) 把它展平成 [B*(L-1)]，与 shift_logits 的第一维对齐，
            # 用于计算 CrossEntropyLoss(shift_logits, shift_labels)。

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    torch.save(model.state_dict(), f'model.pt')


def main():
    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    # 加载模型
    model = ClipCaptionModel().to(device)

    dataset = ClipCapDataset(tokenizer)
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    train(model, train_dataloader, optimizer)


if __name__ == '__main__':
    main()
