import argparse
import numpy as np
import torch
import time
import os
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             matthews_corrcoef, accuracy_score, 
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split, StratifiedKFold
from utils import load_data, load_aligned_features, load_aligned_host_features, similarity_network_fusion, ConstructGraph, get_mp, get_pos
from model import MGNN

np.random.seed(42)
torch.manual_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='MCL-PHI')
    parser.add_argument("--data_path", type=str, default='../../data/', help='Path to dataset')
    
    # 噬菌体外部特征路径
    parser.add_argument("--seq_feat_path", type=str, default='../FCGR+CNN/embeddings_resnet18_phage', help="噬菌体序列特征")
    parser.add_argument("--text_feat_path", type=str, default='../LLM+BioBERT/phage_embeddings', help="噬菌体文本特征")
    
    # 新增：宿主外部特征路径
    parser.add_argument("--host_seq_feat_path", type=str, default='../FCGR+CNN/embeddings_resnet18_host', help="宿主序列特征")
    parser.add_argument("--host_text_feat_path", type=str, default='../LLM+BioBERT/host_embeddings', help="宿主文本特征")
    
    parser.add_argument("--feat_fusion", type=str, default="concat", choices=['concat', 'replace'], help="特征融合方式：concat 或 replace")
    
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--dim_embedding", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument('--cl_weight', type=float, default=0.4)
    parser.add_argument('--snf_k', type=int, default=20)
    parser.add_argument('--snf_t', type=int, default=20)
    parser.add_argument('--feature_type', type=str, default='similarity', help="eye or similarity")
    return parser.parse_args()

def train_and_evaluate(DSAtrain, DSAvalid, DSAtest, phage_feat, host_feat, phage_phage, host_host, phage_host_original, g, args):
    device = torch.device(args.device)
    
    phage_host_label = np.zeros(phage_host_original.shape)
    mask_np = np.zeros(phage_host_original.shape)
    for ele in DSAtrain:
        phage_host_label[ele[0], ele[1]] = ele[2]
        mask_np[ele[0], ele[1]] = 1
    
    phage_host_label_t = torch.FloatTensor(phage_host_label).to(device)
    mask_t = torch.FloatTensor(mask_np).to(device)
    phage_host_T_t = phage_host_label_t.T
    
    phage_phage_t = torch.FloatTensor(phage_phage).to(device)
    host_host_t = torch.FloatTensor(host_host).to(device)
    
    phage_agg_matrixs = [
        phage_phage_t, phage_host_label_t, torch.zeros_like(phage_host_T_t), 
        torch.zeros_like(phage_host_label_t), torch.zeros_like(phage_host_label_t), torch.zeros_like(phage_host_label_t)
    ]
    host_agg_matrixs = [
        host_host_t, phage_host_T_t, torch.zeros_like(phage_host_label_t), torch.zeros_like(host_host_t)
    ]
    
    mps_len_dict = get_mp(phage_phage, host_host, phage_host_original, device)
    mp_len_dict = {k: len(v) for k, v in mps_len_dict.items()}
    pos_dict = get_pos(g, device)

    model = MGNN(d=phage_feat.shape[1], s=host_feat.shape[1], dim=args.dim_embedding, layer=args.layer, g=g,
                 mps_len_dict=mps_len_dict, mp_len_dict=mp_len_dict, cl_weight=args.cl_weight, tau=args.tau)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_valid_aupr = 0.
    best_metrics = {'auc': 0., 'aupr': 0., 'mcc': 0., 'acc': 0., 'pre': 0., 'rec': 0., 'f1': 0.}
    patience_cnt = 0

    for epoch in range(args.epochs):
        model.train()
        predict, loss = model(phage_feat, host_feat, phage_agg_matrixs, host_agg_matrixs, phage_host_label_t, mask_t, pos_dict)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            results = predict.cpu().numpy()
            pred_list_val = [results[ele[0], ele[1]] for ele in DSAvalid]
            gt_val = [ele[2] for ele in DSAvalid]
            valid_aupr = average_precision_score(gt_val, pred_list_val)

            pred_list_test = [results[ele[0], ele[1]] for ele in DSAtest]
            gt_test = [ele[2] for ele in DSAtest]
            
            test_auc = roc_auc_score(gt_test, pred_list_test)
            test_aupr = average_precision_score(gt_test, pred_list_test)
            pred_bin_test = [1 if p >= 0.5 else 0 for p in pred_list_test] # 阈值 0.5
            
            if valid_aupr >= best_valid_aupr:
                best_valid_aupr = valid_aupr
                patience_cnt = 0
                best_metrics.update({
                    'auc': test_auc, 'aupr': test_aupr, 'mcc': matthews_corrcoef(gt_test, pred_bin_test),
                    'acc': accuracy_score(gt_test, pred_bin_test), 'pre': precision_score(gt_test, pred_bin_test, zero_division=0),
                    'rec': recall_score(gt_test, pred_bin_test, zero_division=0), 'f1': f1_score(gt_test, pred_bin_test, zero_division=0)
                })
                print(f"Epoch {epoch:04d} [BEST] | Loss: {loss.item():.4f} | Val AUPR: {valid_aupr:.4f} | Test AUC: {test_auc:.4f}")
            else:
                patience_cnt += 1
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:04d}        | Loss: {loss.item():.4f} | Val AUPR: {valid_aupr:.4f} | 早停倒计时: {patience_cnt}/{args.patience}")
                if patience_cnt > args.patience:
                    print(f"==> 触发早停机制")
                    break
                    
    return list(best_metrics.values())

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # 1. 接收 load_data 返回的 unique_hosts 列表
    phage_phage_1, phage_phage_2, host_host, phage_host_original, unique_hosts = load_data(args.data_path)
    mapping_file = os.path.join(args.data_path, 'phage_similarity_1_metadata_mapping.txt')
    
    # 2. 噬菌体 SNF
    phage_phage_sim = similarity_network_fusion([phage_phage_1, phage_phage_2], K=args.snf_k, t=args.snf_t)
    
    # ==========================
    # 3. 噬菌体特征 (Phage Features)
    # ==========================
    base_phage_feat = torch.from_numpy(phage_phage_sim).float() if args.feature_type == 'similarity' else torch.eye(phage_phage_sim.shape[0], dtype=torch.float)
    
    phage_external_feats = []
    if args.seq_feat_path and os.path.exists(args.seq_feat_path):
        seq_feats = torch.nn.functional.normalize(torch.from_numpy(load_aligned_features(args.seq_feat_path, mapping_file, 'phage_seq')).float(), p=2, dim=1)
        phage_external_feats.append(seq_feats)
    if args.text_feat_path and os.path.exists(args.text_feat_path):
        text_feats = torch.nn.functional.normalize(torch.from_numpy(load_aligned_features(args.text_feat_path, mapping_file, 'phage_text')).float(), p=2, dim=1)
        phage_external_feats.append(text_feats)
        
    phage_feat = torch.cat([base_phage_feat] + phage_external_feats, dim=1) if args.feat_fusion == 'concat' else (torch.cat(phage_external_feats, dim=1) if phage_external_feats else base_phage_feat)
    phage_feat = phage_feat.to(device)
    print(f"最终噬菌体输入维度: {phage_feat.shape}")

    # ==========================
    # 4. 宿主特征 (Host Features)
    # ==========================
    # 基础图特征（单位矩阵）
    base_host_feat = torch.eye(host_host.shape[0], dtype=torch.float)
    host_external_feats = []
    
    # 读取宿主序列特征
    if args.host_seq_feat_path and os.path.exists(args.host_seq_feat_path):
        host_seq_feats = torch.nn.functional.normalize(torch.from_numpy(load_aligned_host_features(args.host_seq_feat_path, unique_hosts, 'host_seq')).float(), p=2, dim=1)
        host_external_feats.append(host_seq_feats)
        
    # 读取宿主文本特征
    if args.host_text_feat_path and os.path.exists(args.host_text_feat_path):
        host_text_feats = torch.nn.functional.normalize(torch.from_numpy(load_aligned_host_features(args.host_text_feat_path, unique_hosts, 'host_text')).float(), p=2, dim=1)
        host_external_feats.append(host_text_feats)

    # 融合宿主特征
    host_feat = torch.cat([base_host_feat] + host_external_feats, dim=1) if args.feat_fusion == 'concat' else (torch.cat(host_external_feats, dim=1) if host_external_feats else base_host_feat)
    host_feat = host_feat.to(device)
    print(f"最终宿主输入维度: {host_feat.shape}")
    
    # 5. 构建正负样本
    pos_indices = np.argwhere(phage_host_original == 1)
    neg_indices = np.argwhere(phage_host_original == 0)
    neg_indices_sampled = neg_indices[np.random.choice(neg_indices.shape[0], size=pos_indices.shape[0], replace=False)]
    
    data_set = np.concatenate([
        np.hstack([pos_indices, np.ones((pos_indices.shape[0], 1))]),
        np.hstack([neg_indices_sampled, np.zeros((neg_indices_sampled.shape[0], 1))])
    ], axis=0).astype(int)
    
    g = ConstructGraph(phage_phage_sim, host_host, phage_host_original)
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    all_scores = {k: [] for k in ['auc', 'aupr', 'mcc', 'acc', 'pre', 'rec', 'f1']}
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(data_set[:, :2], data_set[:, 2])):
        fold = fold_idx + 1
        print(f"\n============= Fold {fold} =============")
        train_data, val_data = train_test_split(data_set[train_idx], test_size=0.05, random_state=1)
        
        metrics = train_and_evaluate(train_data, val_data, data_set[test_idx], phage_feat, host_feat, 
                                     phage_phage_sim, host_host, phage_host_original, g, args)
        
        for k, v in zip(all_scores.keys(), metrics):
            all_scores[k].append(v)
            
        print(f"\nFold {fold} Final Results:")
        for k, v in zip(all_scores.keys(), metrics):
            print(f"  {k.upper()}: {v:.4f}")
        
    print("\n" + "#"*40)
    print("5-Fold Cross Validation Average Results:")
    for k in all_scores.keys():
        print(f"  Mean {k.upper()}: {np.mean(all_scores[k]):.4f} ± {np.std(all_scores[k]):.4f}")
    print("#"*40 + "\n")

if __name__ == "__main__":
    main()