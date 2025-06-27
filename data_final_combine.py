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

def evaluate_results(result_file, result_real_file):
    """评估模型预测结果的准确率"""
    try:
        # 读取预测结果和真实标签
        predictions = {}  # 存储预测结果
        true_labels = {}  # 存储真实标签
        
        # 读取预测结果
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                file_name, pred = line.strip().split('\t')
                # 将预测结果转换为0/1
                pred_value = 0 if pred.lower() == 'fake' else 1
                predictions[file_name] = pred_value
                
        # 读取真实标签
        with open(result_real_file, 'r', encoding='utf-8') as f:
            for line in f:
                file_name, label = line.strip().split('\t')
                # 将真实标签转换为0/1
                true_value = 0 if label.lower() == 'fake' else 1
                true_labels[file_name] = true_value
        
        # 计算统计数据
        total_count = 0
        correct_count = 0
        fake_total = 0
        fake_correct = 0
        true_total = 0
        true_correct = 0
        
        # 遍历所有预测结果
        for file_name in predictions:
            if file_name in true_labels:
                total_count += 1
                pred = predictions[file_name]
                true = true_labels[file_name]
                
                if pred == true:
                    correct_count += 1
                    
                if true == 0:  # 假新闻
                    fake_total += 1
                    if pred == true:
                        fake_correct += 1
                else:  # 真新闻
                    true_total += 1
                    if pred == true:
                        true_correct += 1
        
        # 计算各项准确率
        accuracy = correct_count / total_count if total_count > 0 else 0
        accuracy_fake = fake_correct / fake_total if fake_total > 0 else 0
        accuracy_true = true_correct / true_total if true_total > 0 else 0

        # 计算F1-score
        # 假新闻为负类（0），真新闻为正类（1）
        # 以真新闻为正类计算
        tp = true_correct  # 真新闻预测对了
        fp = fake_total - fake_correct  # 假新闻被误判为真新闻
        fn = true_total - true_correct  # 真新闻被误判为假新闻

        precision_true = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_true = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score_true = 2 * precision_true * recall_true / (precision_true + recall_true) if (precision_true + recall_true) > 0 else 0

        # 以假新闻为正类计算
        tp_fake = fake_correct  # 假新闻预测对了
        fp_fake = true_total - true_correct  # 真新闻被误判为假新闻
        fn_fake = fake_total - fake_correct  # 假新闻被误判为真新闻

        precision_fake = tp_fake / (tp_fake + fp_fake) if (tp_fake + fp_fake) > 0 else 0
        recall_fake = tp_fake / (tp_fake + fn_fake) if (tp_fake + fn_fake) > 0 else 0
        f1_score_fake = 2 * precision_fake * recall_fake / (precision_fake + recall_fake) if (precision_fake + recall_fake) > 0 else 0

        print("\n===== 评估结果 =====")
        print(f"总体准确率 (Accuracy): {accuracy:.4f}")
        print(f"假新闻准确率 (Accuracy_fake): {accuracy_fake:.4f}")
        print(f"真新闻准确率 (Accuracy_true): {accuracy_true:.4f}")
        print(f"假新闻 F1-score: {f1_score_fake:.4f}")
        print(f"真新闻 F1-score: {f1_score_true:.4f}")
        print(f"总样本数: {total_count}")
        print(f"假新闻样本数: {fake_total}")
        print(f"真新闻样本数: {true_total}")
        
    except Exception as e:
        print(f"评估结果时出错: {str(e)}")

def is_file_already_processed(file_name_no_ext, result_file, result_real_file):
    """检查文件是否已经在结果文件中"""
    try:
        # 检查文件是否存在
        if not (Path(result_file).exists() and Path(result_real_file).exists()):
            return False
            
        # 检查result.txt
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith(file_name_no_ext + '\t'):
                    return True
                    
        return False
    except Exception as e:
        print(f"检查文件处理状态时出错: {str(e)}")
        return False

def main():
    # 设置基础目录路径和结果文件路径
    base_dir = Path(r"C:\Users\84772\Desktop\try\twitter_dataset\devset")
    image_dir = base_dir / "images"
    posts_file = base_dir / "posts.txt"
    result_file = Path(r"C:\Users\84772\Desktop\try\result_final_combine.txt")
    result_real_file = Path(r"C:\Users\84772\Desktop\try\result_real_final_combine.txt")

    # 初始化模型
    # nomic-embed-text 的嵌入维度为 768
    model_dim = 768
    fusion_model = TriModalFusionModel(model_dim=model_dim)
    fusion_model.eval()  # 将模型设置为评估模式

    # 如果结果文件不存在，创建空文件
    if not result_file.exists():
        result_file.touch()
    if not result_real_file.exists():
        result_real_file.touch()

    # 加载真实标签数据
    _, file_name_info = load_posts_data(posts_file)

    # 获取所有支持的文件 (.jpg, .gif, .txt)
    supported_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.gif')) + list(image_dir.glob('*.txt'))
    print(f"找到 {len(supported_files)} 个支持的文件在 images 目录下")

    # 处理所有找到的文件
    for file_path in supported_files:
        file_name_no_ext = file_path.stem
        print(f"--- 处理文件: {file_path.name} (ID: {file_name_no_ext}) ---")

        if is_file_already_processed(file_name_no_ext, result_file, result_real_file):
            print(f"文件 {file_name_no_ext} 已经处理过，跳过...")
            continue

        # 1. 获取文本特征
        post_text = file_name_info.get(file_name_no_ext, {}).get("post_text", "")
        text_emb = get_embedding(post_text) if post_text else torch.zeros(model_dim)

        # 2. 获取媒体（图像/URL）特征
        media_emb = torch.zeros(model_dim)
        if file_path.suffix.lower() in ['.jpg', '.gif']:
            media_emb = process_image_file_for_embedding(file_path)
        elif file_path.suffix.lower() == '.txt':
            media_emb = process_txt_file_for_embedding(file_path)

        # 3. 获取情感特征
        sentiment_emb = torch.zeros(model_dim)
        if post_text:
            sentiment_text = get_sentiment_analysis(post_text)
            sentiment_emb = get_embedding(sentiment_text)
            
        # 为模型准备批次维度
        text_emb = text_emb.unsqueeze(0)
        media_emb = media_emb.unsqueeze(0)
        sentiment_emb = sentiment_emb.unsqueeze(0)

        # 4. 模型推理
        with torch.no_grad():
            logits = fusion_model(text_emb, media_emb, sentiment_emb)
            prediction_idx = torch.argmax(logits, dim=1).item()
            model_verdict = "real" if prediction_idx == 1 else "fake"
        
        print(f"模型预测: {model_verdict}")

        # 5. 保存结果
        with open(result_file, 'a', encoding='utf-8') as f_res:
            f_res.write(f"{file_name_no_ext}\t{model_verdict}\n")

        if file_name_no_ext in file_name_info:
            actual_label = file_name_info[file_name_no_ext]["label"]
            with open(result_real_file, 'a', encoding='utf-8') as f_real:
                f_real.write(f"{file_name_no_ext}\t{actual_label}\n")
            print(f"真实标签: {actual_label}")
        else:
            print(f"警告: 在 posts.txt 中未找到文件 {file_name_no_ext} 的标签信息。")
            
        print("-" * 30)

    print("\n===== 处理完成 =====")
    print(f"模型预测结果已保存至: {result_file}")
    print(f"对应的真实标签已保存至: {result_real_file}")
    
    # 添加评估步骤
    evaluate_results(result_file, result_real_file)

if __name__ == "__main__":
    main()

# ===== 处理完成 =====
# 模型预测结果已保存至: C:\Users\84772\Desktop\try\result_final_combine.txt  
# 对应的真实标签已保存至: C:\Users\84772\Desktop\try\result_real_final_combine.txt

# ===== 评估结果 =====
# 总体准确率 (Accuracy): 0.6284
# 假新闻准确率 (Accuracy_fake): 0.4889
# 真新闻准确率 (Accuracy_true): 0.7634
# 假新闻 F1-score: 0.5641
# 真新闻 F1-score: 0.6762
# 总样本数: 366
# 假新闻样本数: 180
# 真新闻样本数: 186