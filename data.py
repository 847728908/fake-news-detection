#只要求模型判断新闻的真伪
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

def is_video_url(url):
    """判断URL是否为视频链接"""
    video_platforms = {
        'youtube.com': r'(?:v=|/v/|/embed/|/watch\?v=)([^&]+)',
        'youtu.be': r'youtu\.be/([^?]+)',
        'bilibili.com': r'video/BV\w+',
        'douyin.com': r'video/\d+',
        'kuaishou.com': r'video/\d+',
    }
    
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    
    for platform, pattern in video_platforms.items():
        if platform in domain:
            return True
    return False

def extract_video_info(url):
    """提取视频相关信息"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 提取标题
        title = soup.title.string if soup.title else "无标题"
        
        # 提取视频描述（如果有）
        description = ""
        # 针对不同平台的描述提取
        meta_desc = soup.find('meta', {'name': 'description'}) or soup.find('meta', {'property': 'og:description'})
        if meta_desc:
            description = meta_desc.get('content', '')
            
        return f"视频标题: {title}\n视频描述: {description}\n视频链接: {url}"
        
    except Exception as e:
        return f"提取视频信息时出错 {url}: {str(e)}"

def extract_web_content(url):
    """从URL中提取网页内容"""
    try:
        # 首先判断是否为视频URL
        if is_video_url(url):
            return extract_video_info(url)
            
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '').lower()
        
        # 处理不同类型的内容
        if 'text/html' in content_type:
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else "无标题"
            
            # 移除script和style元素
            for script in soup(["script", "style"]):
                script.decompose()
                
            text = soup.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = '\n'.join(lines[:20])
            
            return f"网页标题: {title}\n\n主要内容:\n{text}"
            
        elif 'image' in content_type:
            return f"图片内容: {url}"
            
        elif 'video' in content_type:
            return f"视频内容: {url}"
            
        elif 'audio' in content_type:
            return f"音频内容: {url}"
            
        else:
            return f"未知内容类型 ({content_type}): {url}"
            
    except Exception as e:
        return f"提取网页内容时出错 {url}: {str(e)}"

def process_image_file(image_path):
    """为图片文件生成描述性文本"""
    try:
        # 可以在这里添加更复杂的图片分析逻辑，如果模型支持
        return f"这是一张图片，文件名为: {image_path.name}"
    except Exception as e:
        print(f"处理图片文件 {image_path} 时出错: {str(e)}")
        return None

def process_url(url):
    """处理URL链接"""
    try:
        # 检查URL是否有效
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            return f"无效的URL格式: {url}"
            
        # 获取网页内容
        content = extract_web_content(url)
        return f"URL: {url}\n{content}"
        
    except Exception as e:
        print(f"无法访问URL {url}: {str(e)}")
        return None

def process_txt_file_for_url(txt_path):
    """从txt文件中读取URL并处理"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            url = f.readline().strip() # 假设URL在第一行
            if url:
                # 检查URL是否有效
                parsed_url = urlparse(url)
                if not all([parsed_url.scheme, parsed_url.netloc]):
                     print(f"文件 {txt_path} 中的内容不是有效的URL: {url}")
                     return None # 返回None表示处理失败或内容不是URL
                
                print(f"从 {txt_path.name} 中提取URL: {url}")
                return process_url(url) # 调用原有的process_url处理提取到的URL
            else:
                print(f"文件 {txt_path} 为空或第一行不是URL。")
                return None
    except Exception as e:
        print(f"处理文本文件 {txt_path} 时出错: {str(e)}")
        return None

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
                                image_name_to_info[img_name_no_ext] = {"post_id": post_id, "label": label}

        print(f"成功加载了 {len(labels_dict)} 条新闻标签数据 和 {len(image_name_to_info)} 条文件名映射")
        return labels_dict, image_name_to_info
    except Exception as e:
        print(f"加载posts文件时出错: {str(e)}")
        return {}, {}

def interact_with_model(content):
    """与Ollama模型交互，判别新闻真伪"""
    prompt = """请作为一个专业的新闻真伪判别专家，根据以下内容判断这是真新闻还是假新闻。
    
请直接给出明确的判断结果：真新闻或假新闻

需要分析的内容：
{content}

请给出你的分析："""

    try:
        response = ollama.chat(model='deepseek-r1:7b', messages=[{
            'role': 'user',
            'content': prompt.format(content=content)
        }])
        return response['message']['content']
    except Exception as e:
        print(f"模型交互出错: {str(e)}")
        return None

def determine_model_verdict(response):
    """从模型响应中提取判断结果（真/假）"""
    if not response:
        return None
        
    # 寻找包含"真新闻"或"假新闻"的最后一句话
    response = response.lower()
    if "假新闻" in response:
        return "fake"
    elif "真新闻" in response:
        return "real"
    else:
        # 如果没有明确判断，尝试查找其他可能的关键词
        if "虚假" in response or "不可信" in response or "误导" in response:
            return "fake"
        elif "真实" in response or "可信" in response or "准确" in response:
            return "real"
    
    return None  # 无法确定判断结果

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
        
        print("\n===== 评估结果 =====")
        print(f"总体准确率 (Accuracy): {accuracy:.4f}")
        print(f"假新闻准确率 (Accuracy_fake): {accuracy_fake:.4f}")
        print(f"真新闻准确率 (Accuracy_true): {accuracy_true:.4f}")
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
    posts_file = base_dir / "posts.txt" # posts.txt 路径调整
    result_file = Path(r"C:\Users\84772\Desktop\try\result.txt")
    result_real_file = Path(r"C:\Users\84772\Desktop\try\result_real.txt")

    # 如果结果文件不存在，创建空文件
    if not result_file.exists():
        open(result_file, 'w').close()
    if not result_real_file.exists():
        open(result_real_file, 'w').close()

    # 加载真实标签数据
    true_labels, file_name_info = load_posts_data(posts_file) # 重命名变量以清晰

    # 获取所有支持的文件 (.jpg, .gif, .txt)
    supported_files = []
    for ext in ['*.jpg', '*.gif', '*.txt']:
        supported_files.extend(list(image_dir.glob(ext)))

    print(f"找到 {len(supported_files)} 个支持的文件在 images 目录下")

    # 处理所有找到的文件
    for file_path in supported_files:
        file_name_no_ext = file_path.stem # 获取不带扩展名的文件名
        print(f"--- 处理文件: {file_path.name} (ID: {file_name_no_ext}) ---")

        # 检查文件是否已经处理过
        if is_file_already_processed(file_name_no_ext, result_file, result_real_file):
            print(f"文件 {file_name_no_ext} 已经处理过，跳过...")
            continue

        content_for_model = None
        # 根据文件类型处理
        if file_path.suffix.lower() in ['.jpg', '.gif']:
            print("类型: 图片")
            content_for_model = process_image_file(file_path)
        elif file_path.suffix.lower() == '.txt':
            print("类型: 文本 (尝试提取URL)")
            content_for_model = process_txt_file_for_url(file_path) # 使用新的处理函数
        else:
            print(f"跳过不支持的文件类型: {file_path.suffix}")
            continue # 跳过这个文件

        # 如果成功获取到用于模型分析的内容
        if content_for_model:
            print("准备调用大模型分析...")
            response = interact_with_model(content_for_model)
            if response:
                print(f"模型响应: {response[:100]}...\n") # 打印部分响应
                # 提取模型判断结果
                model_verdict = determine_model_verdict(response)

                if model_verdict:
                    # 1. 将 文件名 和 预测结果 写入 result.txt
                    with open(result_file, 'a', encoding='utf-8') as f_res:
                        f_res.write(f"{file_name_no_ext}\t{model_verdict}\n")
                    print(f"写入预测结果: {file_name_no_ext} -> {model_verdict}")

                    # 2. 查找真实标签并写入 result_real.txt
                    if file_name_no_ext in file_name_info:
                        info = file_name_info[file_name_no_ext]
                        actual_label = info["label"]
                        with open(result_real_file, 'a', encoding='utf-8') as f_real:
                            f_real.write(f"{file_name_no_ext}\t{actual_label}\n")
                        print(f"找到并写入真实标签: {file_name_no_ext} -> {actual_label}")
                    else:
                        print(f"警告: 在 posts.txt 中未找到文件 {file_name_no_ext} 的标签信息。")
                else:
                    print("无法从模型响应中确定判断结果。")
            else:
                print("模型交互失败或无响应。")
        else:
            print("无法生成用于模型分析的内容。")
        print("-" * 30) # 分隔符

    print("\n===== 处理完成 =====")
    print(f"模型预测结果已保存至: {result_file}")
    print(f"对应的真实标签已保存至: {result_real_file}")
    
    # 添加评估步骤
    evaluate_results(result_file, result_real_file)

if __name__ == "__main__":
    main()
