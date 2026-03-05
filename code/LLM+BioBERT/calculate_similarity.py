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
embeddings_dir = "phage_embeddings"  # 嵌入向量目录
json_file = os.path.join(embeddings_dir, "embedding_index.json") # 索引文件路径
output_file = "phage_similarity_1.txt"  # 输出的相似度矩阵文件
use_host_embedding = False  # 是否使用宿主预测部分的嵌入
batch_size = 50  # 批处理大小

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
    """加载所有噬菌体的嵌入向量和ID，使用零向量填充缺失的ID"""
    print("加载噬菌体嵌入向量...")
    
    # 1. 扫描所有文件
    embedding_files = glob.glob(os.path.join(embeddings_dir, "*.npz"))
    if not embedding_files:
        embedding_files = glob.glob(os.path.join(embeddings_dir, "*.npy"))
        if not embedding_files:
            # 只要目录存在，允许没有文件（全缺失的情况）
            if not os.path.exists(embeddings_dir):
                raise FileNotFoundError(f"目录不存在: {embeddings_dir}")
            print(f"警告: {embeddings_dir} 中没有找到嵌入文件")
            return np.array([]), []

    # 2. 解析文件映射: ID -> FilePath
    id_to_file = {}
    for fpath in embedding_files:
        fname = os.path.basename(fpath)
        # 匹配 phage_123_embedding 格式
        match = re.search(r'phage_(\d+)_embedding', fname)
        if match:
            pid = int(match.group(1))
            id_to_file[pid] = fpath
        else:
            print(f"跳过无法解析ID的文件: {fname}")

    if not id_to_file:
        print("没有解析到任何有效的噬菌体ID。")
        return np.array([]), []

    # 3. 确定ID范围
    found_ids = sorted(id_to_file.keys())
    min_id = found_ids[0]
    max_id = found_ids[-1]
    
    print(f"检测到ID范围: {min_id} - {max_id}, 实际存在文件数: {len(found_ids)}")
    
    # 4. 确定嵌入维度 (读取第一个有效文件)
    sample_file = id_to_file[min_id]
    try:
        if sample_file.endswith('.npz'):
            data = np.load(sample_file, allow_pickle=True)
            if use_host_embedding and 'host_embedding' in data.files and data['host_embedding'].size > 0:
                sample_emb = data['host_embedding']
            else:
                sample_emb = data['full_embedding']
        else:
            sample_emb = np.load(sample_file)
            
        # 展平
        if sample_emb.ndim > 1: 
            sample_emb = sample_emb.flatten()
        embedding_dim = sample_emb.shape[0]
        print(f"检测到嵌入维度: {embedding_dim}")
    except Exception as e:
        raise RuntimeError(f"无法读取样本文件 {sample_file} 以确定维度: {e}")

    # 5. 构建完整列表 (填充缺失)
    embeddings_list = []
    phage_ids = []
    missing_ids = []
    
    # 假设序列是连续的，填充 min_id 到 max_id 之间的所有空缺
    # 如果您确信应该从 1 开始，可以将 start_id 硬编码为 1
    # start_id = 1 
    start_id = min_id
    full_range = range(start_id, max_id + 1)
    
    print(f"正在构建矩阵 (范围 {start_id} 到 {max_id})...")
    
    for pid in tqdm(full_range, desc="Processing"):
        str_pid = str(pid)
        phage_ids.append(str_pid)
        
        if pid in id_to_file:
            # --- 加载现有文件 ---
            fpath = id_to_file[pid]
            try:
                if fpath.endswith('.npz'):
                    data = np.load(fpath, allow_pickle=True)
                    if use_host_embedding and 'host_embedding' in data.files and data['host_embedding'].size > 0:
                        emb = data['host_embedding']
                    else:
                        emb = data['full_embedding']
                else:
                    emb = np.load(fpath)
                
                # 确保是1D向量
                if emb.ndim > 1: 
                    if emb.shape[0] == 1: emb = emb[0]
                    else: emb = emb.flatten()
                
                # 检查维度一致性
                if emb.shape[0] != embedding_dim:
                    print(f"警告: ID {pid} 维度不匹配 ({emb.shape[0]} vs {embedding_dim}), 填充零向量")
                    embeddings_list.append(np.zeros(embedding_dim, dtype=np.float32))
                    missing_ids.append(pid)
                else:
                    embeddings_list.append(emb)
                    
            except Exception as e:
                print(f"读取文件 {fpath} 失败: {e}, 填充零向量")
                embeddings_list.append(np.zeros(embedding_dim, dtype=np.float32))
                missing_ids.append(pid)
        else:
            # --- 缺失文件，填充零向量 ---
            # print(f"ID {pid} 缺失，填充零向量") # 调试时可开启
            embeddings_list.append(np.zeros(embedding_dim, dtype=np.float32))
            missing_ids.append(pid)
            
    if missing_ids:
        print(f"\n统计: 共填充了 {len(missing_ids)} 个缺失的噬菌体ID (使用零向量)")
        if len(missing_ids) < 20:
            print(f"缺失ID: {missing_ids}")
        else:
            print(f"缺失ID示例: {missing_ids[:10]} ...")

    return np.array(embeddings_list), phage_ids

def compute_similarity_matrix(embeddings, phage_ids):
    """计算相似度矩阵"""
    if len(embeddings) == 0:
        return np.array([])

    print("\n计算相似度矩阵...")
    
    n = len(embeddings)
    similarity_matrix = np.zeros((n, n), dtype=np.float32)
    
    # 批处理计算相似度，避免内存问题
    num_batches = (n + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Computing Batches"):
        start_i = i * batch_size
        end_i = min((i + 1) * batch_size, n)
        batch_embeddings = embeddings[start_i:end_i]
        
        # 计算当前批次与所有嵌入的相似度
        # 注意：零向量与其他向量的 cosine similarity 为 0 (sklearn 处理)
        batch_similarities = cosine_similarity(batch_embeddings, embeddings)
        
        similarity_matrix[start_i:end_i, :] = batch_similarities
        
        # 内存回收
        del batch_similarities
        if i % 10 == 0: 
            import gc
            gc.collect()
    
    # 确保对角线上的值为1.0
    # 注意：如果某个向量是全0，cosine_similarity(0,0) 通常是 0 (因为分母为0)
    # 这里的 fill_diagonal 会强制将其设为 1.0，这对于自身相似度是合理的
    np.fill_diagonal(similarity_matrix, 1.0)
    
    print(f"相似度矩阵计算完成，维度: {similarity_matrix.shape}")
    
    return similarity_matrix

def save_similarity_matrix(similarity_matrix, phage_ids, output_file, metadata=None):
    """保存相似度矩阵及元数据映射文件"""
    if similarity_matrix.size == 0:
        print("相似度矩阵为空，跳过保存。")
        return

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
    
    # 3. 保存详细的元数据映射文件
    mapping_file = output_file.replace('.txt', '_metadata_mapping.txt')
    print(f"  保存元数据映射文件到 {mapping_file}...")
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write("Matrix_Index\tPhage_ID\tPhage_Name\tNCBI_Accession\n")
        
        for idx, pid in enumerate(phage_ids):
            # 尝试从元数据中获取信息
            info = metadata.get(pid, {})
            name = info.get("噬菌体名称", "Unknown/Missing")
            ncbi = info.get("NCBI号", "Unknown/Missing")
            
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
    # 排除对角线
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    off_diagonal = similarity_matrix[mask]
    
    stats = {
        "matrix_size": similarity_matrix.shape,
        "num_phages": len(phage_ids),
        "phage_id_range": [phage_ids[0], phage_ids[-1]],
        "statistics": {
            "mean": float(np.mean(off_diagonal)) if len(off_diagonal) > 0 else 0.0,
            "std": float(np.std(off_diagonal)) if len(off_diagonal) > 0 else 0.0,
            "min": float(np.min(off_diagonal)) if len(off_diagonal) > 0 else 0.0,
            "max": float(np.max(off_diagonal)) if len(off_diagonal) > 0 else 0.0,
            "median": float(np.median(off_diagonal)) if len(off_diagonal) > 0 else 0.0,
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
    print("噬菌体相似度矩阵计算工具 (自动填充缺失ID)")
    print("=" * 70)
    
    # 0. 加载元数据
    metadata = load_metadata_json(json_file)
    
    # 1. 加载嵌入向量
    embeddings, phage_ids = load_embeddings()
    
    if len(embeddings) > 0:
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
            count = 0
            # 尝试找非零向量的样本，避免全是0vs0
            for _ in range(20): # 最多尝试20次
                if count >= 5: break
                i, j = np.random.randint(0, n, 2)
                if i != j:
                    pid_i = phage_ids[i]
                    pid_j = phage_ids[j]
                    name_i = metadata.get(pid_i, {}).get("噬菌体名称", pid_i)
                    name_j = metadata.get(pid_j, {}).get("噬菌体名称", pid_j)
                    
                    name_i = (name_i[:20] + '..') if len(name_i) > 20 else name_i
                    name_j = (name_j[:20] + '..') if len(name_j) > 20 else name_j
                    
                    print(f"  {name_i:<25} vs {name_j:<25}: {similarity_matrix[i, j]:.4f}")
                    count += 1
    else:
        print("未处理任何数据。")

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"✓ 计算完成! 总耗时: {elapsed_time:.2f} 秒")
    print("=" * 70)

if __name__ == "__main__":
    main()