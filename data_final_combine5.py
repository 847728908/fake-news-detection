import os
import requests
from pathlib import Path
import pandas as pd
from PIL import Image
import ollama
import glob
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import time
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# --- Begin: MCAN-style Fusion Model Definition ---

class MultimodalAttention(nn.Module):
    """
    Dot-product attention mechanism.
    Treats the feature vector as a sequence.
    """
    def __init__(self, attention_dropout=0.1):
        super(MultimodalAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        
        self.linear_k = nn.Linear(model_dim, model_dim)
        self.linear_v = nn.Linear(model_dim, model_dim)
        self.linear_q = nn.Linear(model_dim, model_dim)

        self.dot_product_attention = MultimodalAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        residual = query
        batch_size = key.size(0)

        # Reshape for multi-head attention
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        
        # Unsqueeze for bmm
        key = key.reshape(-1, key.size(-2), key.size(-1))
        value = value.reshape(-1, value.size(-2), value.size(-1))
        query = query.reshape(-1, query.size(-2), query.size(-1))

        scale = (key.size(-1) // self.num_heads)**-0.5
        context = self.dot_product_attention(query, key, value, scale, attn_mask)
        
        context = context.view(batch_size, self.num_heads, -1, self.dim_per_head).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.num_heads * self.dim_per_head)
        
        output = self.linear_final(context)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)
        return output

class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=768, ffn_dim=2048, dropout=0.1):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x
        x = self.w2(F.relu(self.w1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x

class TriModalFusionLayer(nn.Module):
    """A layer for fusing three modalities."""
    def __init__(self, model_dim=768, num_heads=8, ffn_dim=2048, dropout=0.1):
        super(TriModalFusionLayer, self).__init__()
        # Attention modules for cross-modal interactions
        self.attn_t_i = MultiHeadAttention(model_dim, num_heads, dropout)
        self.attn_t_s = MultiHeadAttention(model_dim, num_heads, dropout)
        self.attn_i_t = MultiHeadAttention(model_dim, num_heads, dropout)
        self.attn_i_s = MultiHeadAttention(model_dim, num_heads, dropout)
        self.attn_s_t = MultiHeadAttention(model_dim, num_heads, dropout)
        self.attn_s_i = MultiHeadAttention(model_dim, num_heads, dropout)

        # Feed-forward networks for each modality
        self.ffn_t = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        self.ffn_i = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        self.ffn_s = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, text_feat, image_feat, sentiment_feat, attn_mask=None):
        # Update text features with info from image and sentiment
        text_from_img = self.attn_t_i(image_feat, image_feat, text_feat, attn_mask)
        text_from_senti = self.attn_t_s(sentiment_feat, sentiment_feat, text_feat, attn_mask)
        text_output = self.ffn_t(text_from_img + text_from_senti)

        # Update image features with info from text and sentiment
        img_from_text = self.attn_i_t(text_feat, text_feat, image_feat, attn_mask)
        img_from_senti = self.attn_i_s(sentiment_feat, sentiment_feat, image_feat, attn_mask)
        image_output = self.ffn_i(img_from_text + img_from_senti)

        # Update sentiment features with info from text and image
        senti_from_text = self.attn_s_t(text_feat, text_feat, sentiment_feat, attn_mask)
        senti_from_img = self.attn_s_i(image_feat, image_feat, sentiment_feat, attn_mask)
        sentiment_output = self.ffn_s(senti_from_text + senti_from_img)
        
        return text_output, image_output, sentiment_output

class TriModalFusionModel(nn.Module):
    """
    The main model that stacks fusion layers and classifies the result.
    """
    def __init__(self, num_layers=4, model_dim=768, num_heads=8, ffn_dim=2048, dropout=0.1):
        super(TriModalFusionModel, self).__init__()
        self.model_dim = model_dim
        
        self.fusion_layers = nn.ModuleList(
            [TriModalFusionLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        
        self.final_fusion_linear = nn.Linear(model_dim * 3, model_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 2)  # real/fake
        )

    def forward(self, text_emb, image_emb, sentiment_emb):
        # Unsqueeze to add sequence dimension
        text_emb = text_emb.unsqueeze(1)
        image_emb = image_emb.unsqueeze(1)
        sentiment_emb = sentiment_emb.unsqueeze(1)
        
        text_out, img_out, senti_out = text_emb, image_emb, sentiment_emb
        for layer in self.fusion_layers:
            text_out, img_out, senti_out = layer(text_out, img_out, senti_out)
        
        final_representation = torch.cat([text_out, img_out, senti_out], dim=1)
        # Squeeze to remove sequence dimension before linear layer
        final_representation = final_representation.view(final_representation.size(0), -1)
        
        fused_output = F.relu(self.final_fusion_linear(final_representation))
        logits = self.classifier(fused_output)
        return logits

# --- End: MCAN-style Fusion Model Definition ---

def get_embedding(text: str, model: str = "nomic-embed-text") -> torch.Tensor:
    """为给定的文本使用Ollama模型生成嵌入。"""
    try:
        if not text or not text.strip():
            print(f"警告：提供给 get_embedding 的文本为空。返回零张量。")
            # 嵌入维度应与模型输出维度匹配。nomic-embed-text 输出 768。
            return torch.zeros(768)
            
        response = ollama.embeddings(model=model, prompt=text)
        embedding = response['embedding']
        return torch.tensor(embedding, dtype=torch.float32)
    except Exception as e:
        print(f"使用模型 {model} 为文本 '{text[:50]}...' 获取嵌入时出错: {e}")
        return torch.zeros(768) # 错误时返回零向量

def get_sentiment_analysis(text: str) -> str:
    """从语言模型获取文本情感分析。"""
    prompt = f"""分析以下文本的情感。
    描述情感（积极、消极、中性）并简要解释原因。
    文本: "{text}"
    """
    try:
        response = ollama.chat(model='deepseek-r1:7b', messages=[{
            'role': 'user',
            'content': prompt
        }])
        return response['message']['content']
    except Exception as e:
        print(f"获取情感分析时出错: {e}")
        return "中性" # 错误时默认情感文本

def process_image_file_for_embedding(image_path: Path) -> torch.Tensor:
    """通过先创建描述来为图像文件生成嵌入。"""
    try:
        print(f"正在为图片生成描述: {image_path.name}")
        # 使用像llava这样的多模态模型来描述图像。
        response = ollama.chat(
            model='llava:7b',
            messages=[
                {
                    'role': 'user',
                    'content': '为新闻事实核查系统描述这张图片。',
                    'images': [str(image_path)]
                }
            ]
        )
        description = response['message']['content']
        print(f"图片描述: {description[:100]}...")
        return get_embedding(description)
    except Exception as e:
        print(f"处理图像文件 {image_path} 时出错: {e}")
        return torch.zeros(768)

def extract_web_text(url: str) -> str:
    """从URL中提取纯净的文本内容。"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' in content_type:
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator='\n', strip=True)
            return text
        else:
            print(f"URL {url} 不是HTML内容，跳过。")
            return ""
    except Exception as e:
        print(f"从 {url} 提取网页文本时出错: {e}")
        return ""

def process_url_for_embedding(url: str) -> torch.Tensor:
    """处理URL以获取其内容的嵌入。"""
    try:
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            print(f"无效的URL格式: {url}")
            return torch.zeros(768)
        
        content = extract_web_text(url)
        if content:
            return get_embedding(content)
        else:
            return torch.zeros(768)
    except Exception as e:
        print(f"处理URL {url} 时出错: {e}")
        return torch.zeros(768)

def process_txt_file_for_embedding(txt_path: Path) -> torch.Tensor:
    """从 .txt 文件中读取URL并为其内容获取嵌入。"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            url = f.readline().strip()
            if url:
                return process_url_for_embedding(url)
            else:
                print(f"文件 {txt_path} 为空。")
                return torch.zeros(768)
    except Exception as e:
        print(f"处理txt文件 {txt_path} 时出错: {e}")
        return torch.zeros(768)

def load_posts_data(posts_file):
    """加载包含新闻真假标签的文件，确保 image_name_to_info 使用无后缀文件名作为键。"""
    labels_dict = {}
    image_name_to_info = {} # 存储文件名(无后缀)到 post_id 和 label 的映射
    try:
        with open(posts_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                return {}, {}

            # 跳过标题行
            for line in lines[1:]:
                parts = line.strip().split('\t')
                if len(parts) >= 7:
                    post_id = parts[0]
                    post_text = parts[1]  # 第二列
                    image_ids_str = parts[3]  # 修改这里：使用索引3来获取image_id(s)字段
                    label = parts[6]
                    labels_dict[post_id] = label
                    print(f"!!!!!!!image_ids_str: {image_ids_str}, label: {label}!!!!!")
                    if image_ids_str and image_ids_str.lower() != 'nan' and image_ids_str.strip():
                        image_ids = [img_id.strip() for img_id in image_ids_str.split(',')]
                        for img_name_with_ext in image_ids:
                            if img_name_with_ext:
                                # 使用文件名（不含后缀）作为键
                                img_name_no_ext = Path(img_name_with_ext).stem
                                image_name_to_info[img_name_no_ext] = {
                                    "post_id": post_id,
                                    "label": label,
                                    "post_text": post_text  # 新增
                                }

        print(f"成功加载了 {len(labels_dict)} 条新闻标签数据 和 {len(image_name_to_info)} 条文件名映射")
        return labels_dict, image_name_to_info
    except Exception as e:
        print(f"加载posts文件时出错: {str(e)}")
        return {}, {}

def preprocess_and_save_data(base_dir, num_train, num_val, num_test):
    """
    对数据进行预处理：提取特征并保存为文件。
    """
    print("===== 开始特征预处理... =====")
    image_dir = base_dir / "images"
    posts_file = base_dir / "posts.txt"
    model_dim = 768

    _, file_name_info = load_posts_data(posts_file)
    
    supported_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.gif')) + list(image_dir.glob('*.txt'))
    
    labeled_files = [f for f in supported_files if f.stem in file_name_info]
    print(f"找到 {len(labeled_files)} 个带标签的可用文件。")
    
    if len(labeled_files) < (num_train + num_val + num_test):
        raise ValueError(f"可用文件数量 {len(labeled_files)} 不足以满足需求: "
                         f"{num_train} 训练, {num_val} 验证, {num_test} 测试。")

    # 为了可复现性，对文件进行排序
    labeled_files.sort(key=lambda x: x.name)

    # 划分数据集
    train_files = labeled_files[:num_train]
    val_files = labeled_files[num_train : num_train + num_val]
    test_files = labeled_files[num_train + num_val : num_train + num_val + num_test]

    datasets = {'train': train_files, 'validation': val_files, 'test': test_files}

    for name, files in datasets.items():
        print(f"\n--- 正在处理 {name} 数据集 ({len(files)} 条) ---")
        
        all_text_emb, all_media_emb, all_sentiment_emb, all_labels = [], [], [], []

        for file_path in files:
            file_name_no_ext = file_path.stem
            print(f"处理文件: {file_path.name}")
            
            info = file_name_info.get(file_name_no_ext, {})
            post_text = info.get("post_text", "")
            label_str = info.get("label", "fake")
            label = 1 if label_str.lower() == 'real' else 0

            # 1. 文本特征
            text_emb = get_embedding(post_text) if post_text else torch.zeros(model_dim)
            
            # 2. 媒体特征
            if file_path.suffix.lower() in ['.jpg', '.gif']:
                media_emb = process_image_file_for_embedding(file_path)
            elif file_path.suffix.lower() == '.txt':
                media_emb = process_txt_file_for_embedding(file_path)
            else:
                media_emb = torch.zeros(model_dim)
            
            # 3. 情感特征
            if post_text:
                sentiment_text = get_sentiment_analysis(post_text)
                sentiment_emb = get_embedding(sentiment_text)
            else:
                sentiment_emb = torch.zeros(model_dim)

            all_text_emb.append(text_emb)
            all_media_emb.append(media_emb)
            all_sentiment_emb.append(sentiment_emb)
            all_labels.append(label)

        # 转换为Tensor并保存
        data_to_save = {
            'text_features': torch.stack(all_text_emb),
            'media_features': torch.stack(all_media_emb),
            'sentiment_features': torch.stack(all_sentiment_emb),
            'labels': torch.tensor(all_labels, dtype=torch.long)
        }
        
        save_path = Path(f'./{name}_data2.pt')
        torch.save(data_to_save, save_path)
        print(f"{name} 数据集特征已保存至 {save_path}")

    print("===== 特征预处理完成 =====")

def evaluate_model(model, data_loader, loss_fn, device):
    """在给定的数据集上评估模型性能"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for (text_emb, media_emb, sentiment_emb, labels) in data_loader:
            text_emb = text_emb.to(device)
            media_emb = media_emb.to(device)
            sentiment_emb = sentiment_emb.to(device)
            labels = labels.to(device)

            logits = model(text_emb, media_emb, sentiment_emb)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1, precision, recall

def main():
    # --- 配置参数 ---
    base_dir = Path(r"C:\Users\84772\Desktop\try\twitter_dataset\devset")
    train_path = Path('./train_data2.pt')
    val_path = Path('./validation_data2.pt')
    test_path = Path('./test_data2.pt')
    
    # 数据集大小
    NUM_TRAIN = 100
    NUM_VAL = 20
    NUM_TEST = 20
    
    # 训练超参数
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4
    MODEL_DIM = 768
    
    # --- 1. 数据预处理 ---
    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        print("未找到预处理好的特征文件，正在生成...")
        preprocess_and_save_data(base_dir, NUM_TRAIN, NUM_VAL, NUM_TEST)
    else:
        print("发现预处理好的特征文件，直接加载。")

    # --- 2. 加载数据并创建DataLoader ---
    train_data = torch.load(train_path)
    val_data = torch.load(val_path)
    test_data = torch.load(test_path)

    train_dataset = TensorDataset(train_data['text_features'], train_data['media_features'], train_data['sentiment_features'], train_data['labels'])
    val_dataset = TensorDataset(val_data['text_features'], val_data['media_features'], val_data['sentiment_features'], val_data['labels'])
    test_dataset = TensorDataset(test_data['text_features'], test_data['media_features'], test_data['sentiment_features'], test_data['labels'])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # --- 3. 初始化模型、损失函数和优化器 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    fusion_model = TriModalFusionModel(model_dim=MODEL_DIM).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=LEARNING_RATE)

    # --- 4. 训练与验证循环 ---
    best_val_accuracy = 0.0
    best_model_path = Path('best_fusion_model2.pth')

    print("\n===== 开始模型训练 =====")
    for epoch in range(NUM_EPOCHS):
        fusion_model.train()
        total_train_loss = 0
        for batch in train_loader:
            text_emb, media_emb, sentiment_emb, labels = [b.to(device) for b in batch]
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            logits = fusion_model(text_emb, media_emb, sentiment_emb)
            
            # 计算损失
            loss = loss_fn(logits, labels)
            total_train_loss += loss.item()
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # 在验证集上评估
        val_loss, val_acc, val_f1, _, _ = evaluate_model(fusion_model, val_loader, loss_fn, device)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # 保存最佳模型
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(fusion_model.state_dict(), best_model_path)
            print(f"  -> 发现新的最佳模型，已保存至 {best_model_path} (Accuracy: {val_acc:.4f})")

    print("===== 训练完成 =====")

    # --- 5. 在测试集上进行最终评估 ---
    print(f"\n===== 加载最佳模型并在测试集上评估 =====")
    fusion_model.load_state_dict(torch.load(best_model_path))
    
    test_loss, test_acc, test_f1, test_precision, test_recall = evaluate_model(fusion_model, test_loader, loss_fn, device)

    print("\n--- 测试集最终评估结果 ---")
    print(f"测试集损失 (Loss): {test_loss:.4f}")
    print(f"测试集准确率 (Accuracy): {test_acc:.4f}")
    print(f"测试集 F1-Score (weighted): {test_f1:.4f}")
    print(f"测试集精确率 (Precision, weighted): {test_precision:.4f}")
    print(f"测试集召回率 (Recall, weighted): {test_recall:.4f}")

if __name__ == "__main__":
    main()

# 数据集划分
