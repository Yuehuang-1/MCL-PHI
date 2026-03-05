import os
import glob
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json
import re

# ======================
# 配置参数
# ======================
# 输入目录：lcel_chain生成的噬菌体分析结果
phage_text_dir = "phage_reports"  # 噬菌体文本信息目录
output_base_dir = "phage_embeddings"  # 输出目录
model_path = "biobert-v1.1"  # BioBERT模型路径

# 确保输出目录存在
os.makedirs(output_base_dir, exist_ok=True)

# ======================
# 嵌入生成工具函数
# ======================
# 函数：平均池化最后一层的隐藏状态，获得句子嵌入
def mean_pooling(model_output, attention_mask):
    # 首先，选择最后一个隐藏层的输出
    token_embeddings = model_output[0]
    
    # 为每个token创建一个attention mask (shape: batch_size x sequence_length)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    # 将mask应用到token嵌入上
    masked_embeddings = token_embeddings * input_mask_expanded
    
    # 对每个句子的token进行求和，然后除以非padding token的数量
    sum_embeddings = torch.sum(masked_embeddings, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # 得到句子嵌入
    return sum_embeddings / sum_mask

# 函数：从噬菌体分析文本中提取结构化信息
def extract_phage_analysis(text):
    """
    从lcel_chain.py生成的噬菌体分析文本中提取结构化信息
    """
    # 初始化结构化数据
    structured_data = {
        "identification": "",
        "structural_analysis": "",
        "protein_function": "",
        "host_prediction": "",
        "applications": "",
        "full_text": text.strip()
    }
    
    # 提取各个分析部分
    identification_pattern = r"Identification:.*?(?=Structural Analysis:|$)"
    structural_pattern = r"Structural Analysis:.*?(?=Protein Function:|$)"
    protein_pattern = r"Protein Function:.*?(?=Host Prediction:|$)"
    host_pattern = r"Host Prediction:.*?(?=Applications:|$)"
    applications_pattern = r"Applications:.*?(?=$)"
    
    # 使用非贪婪匹配模式，提取每个部分的内容
    for section, pattern in [
        ("identification", identification_pattern),
        ("structural_analysis", structural_pattern),
        ("protein_function", protein_pattern),
        ("host_prediction", host_pattern),
        ("applications", applications_pattern)
    ]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            structured_data[section] = match.group(0).strip()
    
    return structured_data

# 函数：生成噬菌体嵌入
def generate_phage_embedding(phage_data, tokenizer, model, device):
    """
    为噬菌体文本生成嵌入表示
    """
    # 提取全文和宿主预测部分
    text = phage_data["full_text"]
    host_text = phage_data.get("host_prediction", "")
    
    # 编码文本
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    # 生成嵌入
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # 通过平均池化获取句子嵌入
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    # 如果存在宿主预测文本，单独生成宿主嵌入
    host_embedding = None
    if host_text:
        host_encoded = tokenizer(host_text, padding=True, truncation=True, max_length=256, return_tensors='pt')
        host_encoded = {k: v.to(device) for k, v in host_encoded.items()}
        
        with torch.no_grad():
            host_output = model(**host_encoded)
        
        host_embedding = mean_pooling(host_output, host_encoded['attention_mask'])
    
    return {
        "full_embedding": sentence_embeddings.cpu().numpy(),
        "host_embedding": host_embedding.cpu().numpy() if host_embedding is not None else None
    }

# ======================
# 主要执行流程
# ======================
# 初始化BioBERT模型和tokenizer
print("正在加载BioBERT模型和tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    
    # 如果有GPU可用，将模型移到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = model.to(device)
    
    # 设置为评估模式
    model.eval()
    
    print("模型加载成功!")
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    exit(1)

# 获取所有噬菌体文本文件
input_files = glob.glob(os.path.join(phage_text_dir, "phage_*.txt"))
total_files = len(input_files)
print(f"找到 {total_files} 个噬菌体文件需要处理")

# 状态跟踪
processed_count = 0
failed_count = 0
failed_files = []

# 处理每个文件
for file_path in tqdm(input_files, desc="处理噬菌体文件"):
    try:
        # 获取文件名和输出路径
        file_name = os.path.basename(file_path)
        phage_id = file_name.replace("phage_", "").replace(".txt", "")
        output_path = os.path.join(output_base_dir, f"phage_{phage_id}_embedding.npz")
        
        # 如果已经处理过，跳过
        if os.path.exists(output_path):
            print(f"文件 {file_name} 已处理，跳过")
            processed_count += 1
            continue
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取ID和NCBI号信息
        phage_metadata = {}
        for line in content.split('\n')[:4]:  # 查看前几行获取元数据
            if ':' in line:
                key, value = line.split(':', 1)
                phage_metadata[key.strip()] = value.strip()
        
        # 提取分析文本内容（从元数据后开始）
        analysis_text = '\n'.join(content.split('\n')[4:])
        
        # 提取结构化信息
        structured_data = extract_phage_analysis(analysis_text)
        
        # 生成嵌入
        embeddings = generate_phage_embedding(structured_data, tokenizer, model, device)
        
        # 保存嵌入和元数据
        np.savez(
            output_path,
            full_embedding=embeddings["full_embedding"],
            host_embedding=embeddings["host_embedding"] if embeddings["host_embedding"] is not None else np.array([]),
            phage_id=phage_id,
            **phage_metadata
        )
        
        processed_count += 1
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        failed_count += 1
        failed_files.append(file_path)
        
    # 添加短暂延迟，避免GPU过载
    time.sleep(0.05)

# 打印处理统计
print(f"\n处理完成!")
print(f"成功处理: {processed_count}/{total_files} 文件")
print(f"处理失败: {failed_count}/{total_files} 文件")

# 保存失败文件列表
if failed_files:
    with open(os.path.join(output_base_dir, "failed_files.json"), 'w') as f:
        json.dump(failed_files, f, indent=2)
    print(f"失败文件列表已保存到 {os.path.join(output_base_dir, 'failed_files.json')}")

# 保存嵌入向量元数据索引
try:
    embedding_files = glob.glob(os.path.join(output_base_dir, "*.npz"))
    embedding_metadata = {}
    
    for emb_file in embedding_files:
        phage_id = os.path.basename(emb_file).replace("phage_", "").replace("_embedding.npz", "")
        npz_data = np.load(emb_file, allow_pickle=True)
        
        # 提取元数据
        metadata = {
            "phage_id": phage_id,
            "embedding_file": os.path.basename(emb_file)
        }
        
        # 添加其他可用的元数据
        for key in npz_data.files:
            if key not in ["full_embedding", "host_embedding"]:
                try:
                    value = str(npz_data[key])
                    if len(value) < 100:  # 避免保存过长的值
                        metadata[key] = value
                except:
                    pass
        
        embedding_metadata[phage_id] = metadata
    
    # 保存元数据索引
    with open(os.path.join(output_base_dir, "embedding_index.json"), 'w') as f:
        json.dump(embedding_metadata, f, indent=2)
    
    print(f"嵌入向量索引已保存到 {os.path.join(output_base_dir, 'embedding_index.json')}")
except Exception as e:
    print(f"保存嵌入向量索引时出错: {str(e)}")