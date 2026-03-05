import numpy as np
import dgl
import torch
import scipy.sparse as sp
import pandas as pd
import os
import json
import re
from sklearn.neighbors import NearestNeighbors

def load_data(data_path):
    print(f"正在从 {data_path} 加载数据...")

    # 1. 加载基准元数据 (Mapping 1)
    map1_path = os.path.join(data_path, 'phage_similarity_1_metadata_mapping.txt')
    df_map1 = pd.read_csv(map1_path, sep='\t')
    
    df_map1['Phage_Name'] = df_map1['Phage_Name'].astype(str).str.strip()
    df_map1['NCBI_Accession'] = df_map1['NCBI_Accession'].astype(str).str.strip()
    df_map1['Clean_Accession'] = df_map1['NCBI_Accession'].apply(lambda x: x.split('.')[0])
    
    name_to_idx_1 = dict(zip(df_map1['Phage_Name'], df_map1['Matrix_Index']))
    acc_to_idx_1 = dict(zip(df_map1['Clean_Accession'], df_map1['Matrix_Index']))
    num_phages = len(df_map1)
    
    # 加载矩阵1
    sim1_path = os.path.join(data_path, 'phage_similarity_1.txt')
    phage_phage_1 = np.loadtxt(sim1_path)

    # 2. 加载并对齐 矩阵2
    map2_path = os.path.join(data_path, 'phage_similarity_2_metadata_mapping.txt')
    sim2_path = os.path.join(data_path, 'phage_similarity_2.txt')
    
    df_map2 = pd.read_csv(map2_path, sep='\t')
    df_map2['Clean_Accession'] = df_map2['NCBI_Accession'].astype(str).str.strip().apply(lambda x: x.split('.')[0])
    df_map2['Phage_Name'] = df_map2['Phage_Name'].astype(str).str.strip()
    
    phage_phage_2_raw = np.loadtxt(sim2_path)
    
    acc_to_idx_2 = dict(zip(df_map2['Clean_Accession'], df_map2['Matrix_Index']))
    name_to_idx_2 = dict(zip(df_map2['Phage_Name'], df_map2['Matrix_Index']))
    
    phage_phage_2 = np.zeros((num_phages, num_phages))
    idx_map = {} 
    
    for acc, idx1 in acc_to_idx_1.items():
        name = df_map1[df_map1['Matrix_Index']==idx1]['Phage_Name'].values[0]
        if acc in acc_to_idx_2:
            idx_map[idx1] = acc_to_idx_2[acc]
        elif name in name_to_idx_2:
            idx_map[idx1] = name_to_idx_2[name]
    
    if len(idx_map) > 0:
        for i in range(num_phages):
            if i in idx_map:
                idx2_row = idx_map[i]
                for j in range(num_phages):
                    if j in idx_map:
                        idx2_col = idx_map[j]
                        phage_phage_2[i, j] = phage_phage_2_raw[idx2_row, idx2_col]
            else:
                phage_phage_2[i, i] = 1.0 

    # 3. 处理 rawA_s.csv 构建关联矩阵
    inter_path = os.path.join(data_path, 'rawA_s.csv')
    df_inter = pd.read_csv(inter_path, header=None, names=['Index', 'Phage_Name', 'Host_Name'])
    
    df_inter['Phage_Name'] = df_inter['Phage_Name'].astype(str).str.strip()
    df_inter['Host_Name'] = df_inter['Host_Name'].astype(str).str.strip()
    
    # 提取并固定宿主的顺序！
    unique_hosts = sorted(df_inter['Host_Name'].unique().tolist())
    host_name_to_idx = {name: i for i, name in enumerate(unique_hosts)}
    num_hosts = len(unique_hosts)
    
    phage_host = np.zeros((num_phages, num_hosts))
    
    for _, row in df_inter.iterrows():
        p_name = row['Phage_Name']
        h_name = row['Host_Name']
        if p_name in name_to_idx_1 and h_name in host_name_to_idx:
            p_idx = name_to_idx_1[p_name]
            h_idx = host_name_to_idx[h_name]
            phage_host[p_idx, h_idx] = 1.0

    # 4. 宿主相似度
    host_host = np.eye(num_hosts)
    
    # 注意这里多返回了一个 unique_hosts 列表
    return phage_phage_1, phage_phage_2, host_host, phage_host, unique_hosts

def load_aligned_features(feature_source, mapping_path, feature_type='seq'):
    """加载噬菌体特征"""
    print(f"[{feature_type}] 正在加载噬菌体特征: {feature_source}")
    df_map = pd.read_csv(mapping_path, sep='\t')
    df_map['Clean_Accession'] = df_map['NCBI_Accession'].astype(str).str.strip().apply(lambda x: x.split('.')[0])
    master_accessions = df_map['Clean_Accession'].tolist()
    num_phages = len(master_accessions)
    
    feature_matrix = []
    missing_count = 0
    sample_dim = 0
    
    index_file = os.path.join(feature_source, "embedding_index.json")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"找不到 {index_file}")
        
    with open(index_file, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
        
    acc_to_file = {}
    for pid, info in index_data.items():
        raw_acc = info.get("NCBI号", "")
        if not raw_acc: continue
        clean_acc = raw_acc.split('.')[0]
        file_name = info.get("embedding_file", "")
        acc_to_file[clean_acc] = os.path.join(feature_source, file_name)
        
    if len(acc_to_file) > 0:
        first_file = list(acc_to_file.values())[0]
        try:
            data = np.load(first_file, allow_pickle=True)
            vec = data['full_embedding'] if first_file.endswith('.npz') and 'full_embedding' in data.files else data
            sample_dim = vec.shape[0] if vec.ndim == 1 else vec.flatten().shape[0]
        except:
            sample_dim = 512 
    
    if sample_dim == 0: sample_dim = 512

    for acc in master_accessions:
        if acc in acc_to_file:
            fpath = acc_to_file[acc]
            try:
                data = np.load(fpath, allow_pickle=True)
                vec = data['full_embedding'] if fpath.endswith('.npz') and 'full_embedding' in data.files else (data['host_embedding'] if fpath.endswith('.npz') else np.load(fpath))
                if vec.ndim > 1: vec = vec.flatten()
                if vec.shape[0] != sample_dim:
                    feature_matrix.append(np.zeros(sample_dim))
                    missing_count += 1
                else:
                    feature_matrix.append(vec)
            except Exception:
                feature_matrix.append(np.zeros(sample_dim))
                missing_count += 1
        else:
            feature_matrix.append(np.zeros(sample_dim))
            missing_count += 1

    return np.array(feature_matrix, dtype=np.float32)


def load_aligned_host_features(feature_source, unique_hosts, feature_type='host_seq'):
    print(f"[{feature_type}] 正在加载宿主特征: {feature_source}")
    num_hosts = len(unique_hosts)
    feature_matrix = []
    missing_count = 0
    sample_dim = 512 

    index_file = os.path.join(feature_source, "embedding_index.json")
    if not os.path.exists(index_file):
        return np.zeros((num_hosts, sample_dim), dtype=np.float32)
        
    with open(index_file, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
        
    name_to_file = {}
    for pid, info in index_data.items():
        file_name = info.get("embedding_file", "")
        if not file_name: continue
        
        # 将 JSON 里的所有值（包括名称、NCBI等）都作为候选匹配词
        candidates = [str(v).strip() for v in info.values() if isinstance(v, (str, int))]
        for c in candidates:
            name_to_file[c] = os.path.join(feature_source, file_name)
            
    if len(name_to_file) > 0:
        first_file = list(name_to_file.values())[0]
        try:
            data = np.load(first_file, allow_pickle=True)
            vec = data['full_embedding'] if first_file.endswith('.npz') and 'full_embedding' in data.files else data
            sample_dim = vec.shape[0] if vec.ndim == 1 else vec.flatten().shape[0]
        except:
            pass

    for host_name in unique_hosts:
        h_name_clean = str(host_name).strip()
        fpath = None
        
        # 1. 精确匹配
        if h_name_clean in name_to_file:
            fpath = name_to_file[h_name_clean]
        else:
            # 2. 模糊/包含匹配
            for k, v in name_to_file.items():
                if len(k) > 3 and (k in h_name_clean or h_name_clean in k):
                    fpath = v
                    break
                    
        if fpath:
            try:
                data = np.load(fpath, allow_pickle=True)
                vec = data['full_embedding'] if fpath.endswith('.npz') and 'full_embedding' in data.files else (data['host_embedding'] if fpath.endswith('.npz') else np.load(fpath))
                if vec.ndim > 1: vec = vec.flatten()
                
                if vec.shape[0] != sample_dim:
                    feature_matrix.append(np.zeros(sample_dim))
                    missing_count += 1
                else:
                    feature_matrix.append(vec)
            except:
                feature_matrix.append(np.zeros(sample_dim))
                missing_count += 1
        else:
            feature_matrix.append(np.zeros(sample_dim))
            missing_count += 1

    return np.array(feature_matrix, dtype=np.float32)

def similarity_network_fusion(networks, K=20, t=20):
    def normalize(W):
        d = np.sum(W, axis=1)
        return W / np.maximum(d[:, None], np.finfo(float).eps)
    
    def process_network(W, K):
        n = W.shape[0]
        np.fill_diagonal(W, 0)
        nbrs = NearestNeighbors(n_neighbors=K+1).fit(W)
        _, indices = nbrs.kneighbors(W)
        W_local = np.zeros_like(W)
        for i in range(n):
            W_local[i, indices[i, 1:]] = W[i, indices[i, 1:]]
            W_local[indices[i, 1:], i] = W[indices[i, 1:], i]
        return normalize(W_local), normalize(W)
    
    n_networks = len(networks)
    n_samples = networks[0].shape[0]
    local_matrices, global_matrices = [], []
    
    for i in range(n_networks):
        W_local, W_global = process_network(networks[i].copy(), K)
        local_matrices.append(W_local)
        global_matrices.append(W_global)
    
    for iteration in range(t):
        new_global_matrices = []
        for i in range(n_networks):
            W_sum = sum([global_matrices[j] for j in range(n_networks) if j != i])
            W_avg = W_sum / (n_networks - 1) if n_networks > 1 else W_sum
            W_new = local_matrices[i] @ W_avg @ local_matrices[i].T
            np.fill_diagonal(W_new, 0)
            new_global_matrices.append(normalize(W_new))
        global_matrices = new_global_matrices
    return sum(global_matrices) / n_networks

def ConstructGraph(phage_phage, host_host, phage_host):
    phage_phage = np.where(phage_phage <= 0.5, 0, phage_phage)
    host_phage = phage_host.T
    src, dst = sp.csr_matrix(phage_phage).nonzero()
    ph_ph = dgl.heterograph({('phage', 'phph', 'phage'): (src, dst)}, num_nodes_dict={'phage': phage_phage.shape[0]})
    src, dst = sp.csr_matrix(host_host).nonzero()
    ho_ho = dgl.heterograph({('host', 'hoho', 'host'): (src, dst)}, num_nodes_dict={'host': host_host.shape[0]})
    src_ph, dst_ph = sp.csr_matrix(phage_host).nonzero()
    ph_ho = dgl.heterograph({('phage', 'phho', 'host'): (src_ph, dst_ph)}, num_nodes_dict={'phage': phage_host.shape[0], 'host': phage_host.shape[1]})
    src_hp, dst_hp = sp.csr_matrix(host_phage).nonzero()
    ho_ph = dgl.heterograph({('host', 'hoph', 'phage'): (src_hp, dst_hp)}, num_nodes_dict={'host': host_phage.shape[0], 'phage': host_phage.shape[1]})
    return dgl.heterograph({
        ('phage', 'phph', 'phage'): ph_ph.edges(),
        ('phage', 'phho', 'host'): ph_ho.edges(),
        ('host', 'hoph', 'phage'): ho_ph.edges(),
        ('host', 'hoho', 'host'): ho_ho.edges()
    }, num_nodes_dict={'phage': phage_phage.shape[0], 'host': host_host.shape[0]})

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_self_loop = sp.diags((1 / (1 + rowsum)).flatten())
    d_inv_sqrt = np.power(rowsum + 1e-8, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt) + d_self_loop).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return torch.sparse.FloatTensor(indices, torch.from_numpy(sparse_mx.data), torch.Size(sparse_mx.shape))

def row_normalize(t):
    if isinstance(t, np.ndarray):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t = torch.from_numpy(t).float().to(device)
    t = t.float()
    row_sums = t.sum(1) + 1e-12
    output = t / row_sums[:, None]
    output[torch.isnan(output) | torch.isinf(output)] = 0.0
    return output

def l2_norm(t, axit=1):
    norm = torch.norm(t.float(), 2, axit, True) + 1e-12
    output = torch.div(t.float(), norm)
    output[torch.isnan(output) | torch.isinf(output)] = 0.0
    return output

def get_mp(phage_phage, host_host, phage_host_original, device):
    phph, hoho, phho = phage_phage, host_host, phage_host_original
    matrices = [
        phph, np.matmul(phho, phho.T) > 0, np.matmul(phph, phph.T) > 0, 
        np.matmul(np.matmul(phph, phph.T) > 0, phph.T) > 0, 
        np.matmul(np.matmul(np.matmul(phph, phph.T) > 0, phph.T) > 0, phph.T) > 0, 
        np.matmul(np.matmul(phho, phho.T) > 0, np.matmul(phho, phho.T) > 0) > 0, 
        hoho, np.matmul(phho.T, phho) > 0, np.matmul(hoho, hoho) > 0, 
        np.matmul(hoho, np.matmul(phho.T, phho) > 0) > 0
    ]
    tensors = [sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(np.array(m, dtype=np.float32)))).to(device) for m in matrices]
    return {'phage': tensors[:6], 'host': tensors[6:]}

def get_pos(g, device):
    host_num, phage_num = g.num_nodes('host'), g.num_nodes('phage')
    host_pos_num, phage_pos_num = min(10, host_num - 1), min(5, phage_num - 1)

    def getMetaPathSrcAndDst(g, metapath):
        adj = 1 
        for etype in metapath: adj = adj * g.adj(etype=etype, scipy_fmt='csr', transpose=False) 
        return adj.tocoo()

    def generate_pos(node_all, node_pos, node_pos_num): 
        for i in range(len(node_all)):
            one = node_all[i].nonzero()[0]
            if len(one) > node_pos_num: 
                sele = one[np.argsort(-node_all[i, one])[:node_pos_num]]
                node_pos[i, sele] = 1
            else: node_pos[i, one] = 1
        return sp.coo_matrix(node_pos)

    hoho = getMetaPathSrcAndDst(g, ['hoho'])
    hoho = hoho / (hoho.sum(axis=-1) + 1e-12).reshape(-1, 1)
    host_pos = generate_pos(sp.coo_matrix(np.identity(host_num)).A.astype("float32"), np.zeros((host_num, host_num)), host_pos_num)

    phph = getMetaPathSrcAndDst(g, ['phph'])
    phph = phph / (phph.sum(axis=-1) + 1e-12).reshape(-1, 1)
    phage_pos = generate_pos(sp.coo_matrix(np.identity(phage_num)).A.astype("float32"), np.zeros((phage_num, phage_num)), phage_pos_num)

    return {'host': sparse_mx_to_torch_sparse_tensor(host_pos).to(device), 'phage': sparse_mx_to_torch_sparse_tensor(phage_pos).to(device)}