import os
import glob
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics.pairwise import cosine_similarity
import json
import re

# ======================
# 配置参数
# ======================
embeddings_dir = "embeddings_pretrained"  # 嵌入向量目录
json_file = os.path.join(embeddings_dir, "embedding_index.json") # 之前生成的索引文件路径
output_file = "phage_similarity_2.txt"  # 输出的相似度矩阵文件
use_host_embedding = False  # 是否使用宿主预测部分的嵌入
batch_size = 50  # 批处理大小，避免内存溢出

# ======================
# 功能函数
# ======================

def load_metadata_json(json_path):
    """加载噬菌体元数据索引文件"""
    print(f"尝试加载元数据索引: {json_path}...")
    if not os.path.exists(json_path):
        print("警告: 找不到 json 索引文件，生成的映射文件将缺失详细名称。")
        return {}
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载 {len(data)} 条噬菌体元数据")
        return data
    except Exception as e:
        print(f"加载 json 失败: {e}")
        return {}

def load_embeddings():
    """加载所有噬菌体的嵌入向量和ID，仅加载实际存在的文件"""
    print("加载噬菌体嵌入向量...")
    
    # 查找所有嵌入文件
    embedding_files = glob.glob(os.path.join(embeddings_dir, "*.npz"))
    
    if not embedding_files:
        embedding_files = glob.glob(os.path.join(embeddings_dir, "*.npy"))
        if not embedding_files:
            raise FileNotFoundError(f"在 {embeddings_dir} 中没有找到嵌入文件")
    
    print(f"找到 {len(embedding_files)} 个嵌入文件")
    
    # 收集嵌入和ID
    embeddings_list = []
    failed_files = []
    
    # 加载所有嵌入
    for file_path in tqdm(embedding_files, desc="读取嵌入文件"):
        file_name = os.path.basename(file_path)
        
        try:
            # 提取序号X
            match = re.search(r'phage_(\d+)_embedding', file_name)
            if match:
                phage_id = int(match.group(1))
            else:
                print(f"无法从文件名 {file_name} 解析序号，跳过")
                continue
            
            if file_path.endswith('.npz'):
                # 对于npz文件(多数组)
                data = np.load(file_path, allow_pickle=True)
                if use_host_embedding and 'host_embedding' in data.files and data['host_embedding'].size > 0:
                    # 使用宿主嵌入
                    embedding = data['host_embedding']
                else:
                    # 使用完整嵌入
                    embedding = data['full_embedding']
            else:
                # 对于npy文件(单数组)
                embedding = np.load(file_path)
            
            # 确保嵌入是2D数组的情况下提取第一行，因为每个文本只有一个嵌入
            if embedding.ndim > 1 and embedding.shape[0] == 1:
                embedding = embedding[0]
            
            # 确保是1D向量
            if embedding.ndim != 1:
                # print(f"警告: {file_name} 的嵌入维度不是1D: {embedding.shape}，尝试展平")
                embedding = embedding.flatten()
            
            # 添加到列表
            embeddings_list.append((phage_id, embedding))
            
        except Exception as e:
            print(f"加载 {file_path} 时出错: {str(e)}，跳过")
            failed_files.append(file_name)
    
    if not embeddings_list:
        raise ValueError("没有成功加载任何嵌入向量！")
    
    # 按序号排序 (这一步非常重要，保证矩阵的行列顺序是固定的)
    embeddings_list.sort(key=lambda x: x[0])
    
    # 分离ID和嵌入
    phage_ids = [str(item[0]) for item in embeddings_list]
    embeddings = np.array([item[1] for item in embeddings_list])
    
    print(f"\n成功加载 {len(embeddings)} 个嵌入向量")
    print(f"噬菌体ID范围: {phage_ids[0]} - {phage_ids[-1]}")
    print(f"嵌入维度: {embeddings.shape[1]}")
    
    if failed_files:
        print(f"\n警告: {len(failed_files)} 个文件加载失败")
    
    return embeddings, phage_ids

def compute_similarity_matrix(embeddings, phage_ids):
    """计算相似度矩阵"""
    print("\n计算相似度矩阵...")
    
    n = len(embeddings)
    similarity_matrix = np.zeros((n, n), dtype=np.float32)
    
    # 批处理计算相似度，避免内存问题
    num_batches = (n + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="计算批次"):
        start_i = i * batch_size
        end_i = min((i + 1) * batch_size, n)
        batch_embeddings = embeddings[start_i:end_i]
        
        # 计算当前批次与所有嵌入的相似度
        batch_similarities = cosine_similarity(batch_embeddings, embeddings)
        
        similarity_matrix[start_i:end_i, :] = batch_similarities
        
        # 避免内存泄漏
        del batch_similarities
        if i % 10 == 0:  # 每10个批次清理一次内存
            import gc
            gc.collect()
    
    # 确保对角线上的值为1.0（自身与自身的相似度）
    np.fill_diagonal(similarity_matrix, 1.0)
    
    print(f"相似度矩阵计算完成，维度: {similarity_matrix.shape}")
    
    return similarity_matrix

def save_similarity_matrix(similarity_matrix, phage_ids, output_file, metadata=None):
    """保存相似度矩阵及元数据映射文件"""
    print(f"\n保存结果文件...")
    
    # 1. 保存为纯相似度矩阵文本格式 (仅数值)
    print(f"  保存纯矩阵格式到 {output_file}...")
    with open(output_file, 'w') as f:
        for i in range(len(phage_ids)):
            row_values = [f"{similarity_matrix[i, j]:.6f}" for j in range(len(phage_ids))]
            f.write('\t'.join(row_values) + '\n')
    
    # 2. 保存包含ID的矩阵版本
    with_ids_file = output_file.replace('.txt', '_with_ids.txt')
    print(f"  保存带ID版本到 {with_ids_file}...")
    with open(with_ids_file, 'w') as f:
        f.write('\t'.join(['Phage_ID'] + phage_ids) + '\n')
        for i, phage_id in enumerate(phage_ids):
            row_values = [phage_id] + [f"{similarity_matrix[i, j]:.6f}" for j in range(len(phage_ids))]
            f.write('\t'.join(row_values) + '\n')
    
    # 3. 保存详细的元数据映射文件 (这是你新要求的)
    mapping_file = output_file.replace('.txt', '_metadata_mapping.txt')
    print(f"  保存元数据映射文件到 {mapping_file}...")
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write("Matrix_Index\tPhage_ID\tPhage_Name\tNCBI_Accession\n")
        
        for idx, pid in enumerate(phage_ids):
            # 尝试从元数据中获取信息
            info = metadata.get(pid, {})
            # 注意：JSON里面的键名是中文，需要对应之前生成的代码
            name = info.get("噬菌体名称", "Unknown")
            ncbi = info.get("NCBI号", "Unknown")
            
            # 写入一行: 矩阵索引(行号)  ID  名称  NCBI号
            f.write(f"{idx}\t{pid}\t{name}\t{ncbi}\n")

    # 4. 保存为压缩的numpy格式
    np_file = output_file.replace('.txt', '.npz')
    print(f"  保存压缩格式到 {np_file}...")
    np.savez_compressed(
        np_file,
        similarity_matrix=similarity_matrix,
        phage_ids=np.array(phage_ids)
    )
    
    # 5. 计算统计信息
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    off_diagonal = similarity_matrix[mask]
    
    stats = {
        "matrix_size": similarity_matrix.shape,
        "num_phages": len(phage_ids),
        "phage_id_range": [phage_ids[0], phage_ids[-1]],
        "statistics": {
            "mean": float(np.mean(off_diagonal)),
            "std": float(np.std(off_diagonal)),
            "min": float(np.min(off_diagonal)),
            "max": float(np.max(off_diagonal)),
            "median": float(np.median(off_diagonal)),
        }
    }
    
    stats_file = output_file.replace('.txt', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ 所有文件已保存。")
    print(f"  详细映射文件: {mapping_file}")

# ======================
# 主执行流程
# ======================
def main():
    start_time = time.time()
    
    print("=" * 70)
    print("噬菌体相似度矩阵计算工具 (含元数据映射)")
    print("=" * 70)
    
    # 0. 加载元数据 (新增)
    metadata = load_metadata_json(json_file)
    
    # 1. 加载嵌入向量
    embeddings, phage_ids = load_embeddings()
    
    # 2. 计算相似度矩阵
    similarity_matrix = compute_similarity_matrix(embeddings, phage_ids)
    
    # 3. 保存相似度矩阵及元数据映射
    save_similarity_matrix(similarity_matrix, phage_ids, output_file, metadata)
    
    # 4. 输出一些相似度样本
    print("\n" + "=" * 70)
    print("相似度样例 (随机选择5对):")
    print("=" * 70)
    n = len(phage_ids)
    if n >= 5:
        np.random.seed(42)
        for _ in range(5):
            i, j = np.random.randint(0, n, 2)
            if i != j:
                pid_i = phage_ids[i]
                pid_j = phage_ids[j]
                # 尝试获取名称
                name_i = metadata.get(pid_i, {}).get("噬菌体名称", pid_i)
                name_j = metadata.get(pid_j, {}).get("噬菌体名称", pid_j)
                
                # 截断太长的名字以便显示
                name_i = (name_i[:20] + '..') if len(name_i) > 20 else name_i
                name_j = (name_j[:20] + '..') if len(name_j) > 20 else name_j
                
                print(f"  {name_i:<25} vs {name_j:<25}: {similarity_matrix[i, j]:.4f}")
    
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"✓ 计算完成! 总耗时: {elapsed_time:.2f} 秒")
    print("=" * 70)

if __name__ == "__main__":
    main()