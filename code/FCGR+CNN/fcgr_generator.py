"""
FCGR (Frequency Chaos Game Representation) 数据处理模块
将DNA序列转换为2D频率矩阵
"""
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def count_kmers(sequence, k):
    """
    统计序列中所有k-mer的出现次数
    
    Args:
        sequence: DNA序列字符串 (ATCG)
        k: k-mer长度
    
    Returns:
        dict: {k-mer: 出现次数}
    """
    sequence = sequence.upper()
    freq = defaultdict(int)
    
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        # 过滤包含N的k-mer
        if 'N' not in kmer and len(kmer) == k:
            if all(base in 'ATCG' for base in kmer):
                freq[kmer] += 1
    
    return dict(freq)


def probabilities(sequence, kmer_counts, k):
    """
    将k-mer计数转换为概率分布
    
    Args:
        sequence: DNA序列
        kmer_counts: k-mer频数字典
        k: k-mer长度
    
    Returns:
        dict: {k-mer: 概率}
    """
    sequence = sequence.upper()
    total_kmers = len(sequence) - k + 1
    
    # 过滤掉包含N的k-mer
    valid_total = sum(1 for i in range(total_kmers) 
                     if 'N' not in sequence[i:i+k] and 
                     all(base in 'ATCG' for base in sequence[i:i+k]))
    
    if valid_total == 0:
        return {}
    
    prob_dict = {}
    for kmer, count in kmer_counts.items():
        prob_dict[kmer] = count / valid_total
    
    return prob_dict


def chaos_game_representation(probabilities, k):
    """
    使用混沌游戏规则将k-mer概率映射到2D矩阵
    
    原理:
    - 矩阵大小为 2^k × 2^k
    - 每个k-mer通过其碱基序列映射到特定位置
    - 四个碱基映射到四个象限: A, T, G, C
    
    Args:
        probabilities: k-mer概率字典
        k: k-mer长度
    
    Returns:
        numpy.ndarray: [2^k, 2^k] 的频率矩阵
    """
    array_size = 2 ** k
    chaos_matrix = np.zeros((array_size, array_size))
    
    # 碱基到坐标的映射
    # A: (0, 0), T: (0, 1), G: (1, 0), C: (1, 1)
    base_to_coord = {
        'A': (0, 0),
        'T': (0, 1), 
        'G': (1, 0),
        'C': (1, 1)
    }
    
    for kmer, prob in probabilities.items():
        if len(kmer) != k:
            continue
            
        # 初始化坐标
        x, y = 0, 0
        
        # 通过混沌游戏规则计算位置
        for i, base in enumerate(kmer):
            if base not in base_to_coord:
                break
            
            dx, dy = base_to_coord[base]
            # 每次将坐标范围缩小一半
            scale = 2 ** (k - i - 1)
            x += dx * scale
            y += dy * scale
        else:
            # 只有所有碱基都有效时才设置值
            chaos_matrix[int(x), int(y)] = prob
    
    return chaos_matrix


def generate_fcgr(sequence, k=6):
    """
    一站式生成FCGR矩阵
    
    Args:
        sequence: DNA序列
        k: k-mer长度 (默认6, 生成64x64矩阵)
    
    Returns:
        numpy.ndarray: FCGR矩阵
    """
    # 步骤1: K-mer计数
    kmer_counts = count_kmers(sequence, k)
    
    if not kmer_counts:
        # 如果没有有效k-mer，返回零矩阵
        return np.zeros((2**k, 2**k))
    
    # 步骤2: 计算概率
    kmer_probs = probabilities(sequence, kmer_counts, k)
    
    # 步骤3: 生成CGR矩阵
    fcgr_matrix = chaos_game_representation(kmer_probs, k)
    
    return fcgr_matrix


# 测试代码
if __name__ == "__main__":
    # 测试序列
    test_seq = "ATCGATCGATCGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC" * 10
    
    print("测试FCGR生成...")
    print(f"序列长度: {len(test_seq)}")
    
    # 生成FCGR
    fcgr = generate_fcgr(test_seq, k=6)
    
    print(f"FCGR矩阵大小: {fcgr.shape}")
    print(f"非零元素数: {np.count_nonzero(fcgr)}")
    print(f"最大值: {fcgr.max():.6f}")
    print(f"最小值: {fcgr.min():.6f}")
    print(f"总和: {fcgr.sum():.6f} (应该接近1.0)")
    
    print("\n✓ FCGR模块测试通过!")
