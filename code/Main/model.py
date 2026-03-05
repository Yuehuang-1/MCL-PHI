import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from utils import row_normalize, l2_norm

# ====================
# 基础组件
# ====================

class BasicAttention(nn.Module):
    def __init__(self, embed_dim):
        super(BasicAttention, self).__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values):
        Q = self.query_proj(queries)
        K = self.key_proj(keys)
        V = self.value_proj(values)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attention_probs = self.softmax(attention_scores)
        return torch.matmul(attention_probs, V)

class SemanticsAttention(nn.Module):
    def __init__(self, in_size, attn_drop=0.5, hidden_size=128):
        super(SemanticsAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.dropout = nn.Dropout(attn_drop)

    def forward(self, z):
        z_stack = torch.stack(z, dim=1) 
        w = self.project(z_stack)
        beta = torch.softmax(w, dim=1)
        return (beta * z_stack).sum(1)

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=128):
        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        self.ouput_dim = input_dim if ouput_dim is None else ouput_dim
        self.W_Q = nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X):
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        return self.fc(context)

# ====================
# 编码器 (Encoders)
# ====================

class GNNFiLM(nn.Module):
    def __init__(self, in_ft, out_ft, num_types, bias=True): 
        super(GNNFiLM, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False) 
        self.act = nn.PReLU() 
        self.fc_gamma = nn.Linear(num_types, out_ft)  
        self.fc_beta = nn.Linear(num_types, out_ft)   
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, node_type):
        seq_fts = self.fc(seq)
        out = torch.spmm(adj, seq_fts)
        node_type_one_hot = F.one_hot(node_type, num_classes=2).float()  
        gamma = self.fc_gamma(node_type_one_hot)
        beta = self.fc_beta(node_type_one_hot)
        out = (gamma * out) + beta
        if self.bias is not None:
            out += self.bias
        out = out + seq_fts 
        return self.act(out)

class MpEncoder(nn.Module):
    def __init__(self, P, hidden_dim, attn_drop, num_types=2): 
        super(MpEncoder, self).__init__()
        self.P = P
        self.node_level = nn.ModuleList([GNNFiLM(hidden_dim, hidden_dim, num_types) for _ in range(P)])
        self.att = SemanticsAttention(hidden_dim, attn_drop)

    def forward(self, h, mps, node_type):
        if self.P == 0: return h
        embeds = []
        for i, mp in enumerate(mps):  
            embeds.append(self.node_level[i](h, mp, node_type))  
        return self.att(embeds) + h

class NeEncoder(nn.Module):
    def __init__(self, out_dim, g, num_heads=4):
        super(NeEncoder, self).__init__()
        self.dim_embedding = out_dim
        
        self.fc_phage_phage = nn.Linear(self.dim_embedding, self.dim_embedding)
        self.fc_phage_host = nn.Linear(self.dim_embedding, self.dim_embedding)
        self.fc_host_phage = nn.Linear(self.dim_embedding, self.dim_embedding)
        self.fc_host_host = nn.Linear(self.dim_embedding, self.dim_embedding)
        
        self.phage_phage_attn = nn.MultiheadAttention(out_dim, num_heads)
        self.phage_host_attn = nn.MultiheadAttention(out_dim, num_heads)
        self.host_phage_attn = nn.MultiheadAttention(out_dim, num_heads)
        self.host_host_attn = nn.MultiheadAttention(out_dim, num_heads)
        
        self.layer_norm = nn.LayerNorm(out_dim)
        
    def forward(self, layer, host_feat, phage_feat, phage_agg_matrixs, host_agg_matrixs):
        device = host_feat.device
        for i in range(layer):
            phage_features = []
            
            # Phage Aggregation
            phage_phage_feat = F.relu(self.fc_phage_phage(phage_feat))
            phage_phage_agg = torch.mm((row_normalize(phage_agg_matrixs[0]).float()).to(device), phage_phage_feat)
            phage_phage_agg, _ = self.phage_phage_attn(phage_phage_agg, phage_phage_agg, phage_phage_agg)
            phage_features.append(phage_phage_agg + phage_feat)
            
            phage_host_feat = F.relu(self.fc_phage_host(phage_feat))
            host_phage_feat = F.relu(self.fc_host_phage(host_feat))
            phage_host_agg = torch.mm((row_normalize(phage_agg_matrixs[1]).float()).to(device), host_phage_feat)
            phage_host_agg, _ = self.phage_host_attn(phage_host_agg, phage_host_agg, phage_host_agg)
            phage_features.append(phage_host_agg + phage_feat)
            
            # Host Aggregation
            host_features = []
            host_host_feat = F.relu(self.fc_host_host(host_feat))
            host_host_agg = torch.mm((row_normalize(host_agg_matrixs[0]).float()).to(device), host_host_feat)
            host_host_agg, _ = self.host_host_attn(host_host_agg, host_host_agg, host_host_agg)
            host_features.append(host_host_agg + host_feat)
            
            host_phage_agg = torch.mm((row_normalize(host_agg_matrixs[1]).float()).to(device), phage_host_feat)
            host_phage_agg, _ = self.host_phage_attn(host_phage_agg, host_phage_agg, host_phage_agg)
            host_features.append(host_phage_agg + host_feat)
            
            phage_features = torch.stack(phage_features, dim=1)
            host_features = torch.stack(host_features, dim=1)
            
            phage_feat = l2_norm(self.layer_norm(torch.mean(phage_features, dim=1)))
            host_feat = l2_norm(self.layer_norm(torch.mean(host_features, dim=1)))
            
        return {'host': host_feat, 'phage': phage_feat}

# ====================
# 对比学习与主模型
# ====================

class Contrast(nn.Module): 
    def __init__(self, out_dim, tau, keys): 
        super(Contrast, self).__init__()
        self.attention = BasicAttention(out_dim)
        self.proj = nn.ModuleDict({k: nn.Sequential( 
            nn.Linear(out_dim, out_dim), nn.ELU(), nn.Linear(out_dim, out_dim)
        ) for k in keys})
        self.tau = tau

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t()) 
        dot_denominator = torch.mm(z1_norm, z2_norm.t()) 
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau) 
        return sim_matrix

    def compute_loss(self, z_mp, z_ne, pos, k): 
        z_proj_mp = self.proj[k](z_mp)   
        z_proj_ne = self.proj[k](z_ne)
        matrix_mp2ne = self.sim(z_proj_mp, z_proj_ne)
        matrix_ne2mp = matrix_mp2ne.t()
        
        softmax_mp2ne = F.softmax(matrix_mp2ne, dim=1)
        lori_mp = -torch.log(softmax_mp2ne.mul(pos.to_dense()).sum(dim=-1)).mean() 
        softmax_ne2mp = F.softmax(matrix_ne2mp, dim=1)
        lori_ne = -torch.log(softmax_ne2mp.mul(pos.to_dense()).sum(dim=-1)).mean() 
        return lori_mp + lori_ne 
    
    def forward(self, z_mp, z_ne, pos):
        sumLoss = 0
        for k, v in pos.items(): 
            sumLoss += self.compute_loss(z_mp[k], z_ne[k], pos[k], k) 
        return sumLoss

class MGNN(nn.Module):
    def __init__(self, d, s, dim, layer, g, mps_len_dict, mp_len_dict, cl_weight, tau):
        super(MGNN, self).__init__()
        self.g = g
        self.layer = layer
        self.dim_embedding = dim
        self.cl_weight = cl_weight
        
        self.ne = NeEncoder(out_dim=dim, g=g)
        self.mps_len_dict = mps_len_dict
        self.mp_len_dict = mp_len_dict

        self.mpencoder = nn.ModuleDict({k: MpEncoder(v, dim, 0.2) for k, v in mp_len_dict.items()})
        self.fc_dict = nn.ModuleDict({k: nn.Linear(dim, dim) for k in ['host', 'phage']})

        self.project_phage_feature = nn.Linear(d, dim)
        self.project_host_feature = nn.Linear(s, dim)

        self.attn = MultiHeadAttention(input_dim=dim * 2, n_heads=3)
        self.contrast = Contrast(dim, tau, ['host', 'phage'])
        self.dropout = nn.Dropout(0.5)

        self.decoder = nn.Sequential(
            nn.Linear(dim * 2, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, phage_feat, host_feat, phage_agg_matrixs, host_agg_matrixs, phage_host, mask, pos_dict):
        node_type_host = torch.zeros(host_feat.shape[0], dtype=torch.long).to(host_feat.device)
        node_type_phage = torch.ones(phage_feat.shape[0], dtype=torch.long).to(phage_feat.device)
        node_type_mapping = {'host': node_type_host, 'phage': node_type_phage}

        phage_feat = self.project_phage_feature(phage_feat)
        host_feat = self.project_host_feature(host_feat)

        node_ne = self.ne(self.layer, host_feat, phage_feat, phage_agg_matrixs, host_agg_matrixs)
        
        node_features = {'host': host_feat, 'phage': phage_feat}
        node_mp = {k: self.mpencoder[k](self.fc_dict[k](node_features[k]), self.mps_len_dict[k], node_type_mapping[k]) 
                   for k in self.mps_len_dict}

        host_fused = torch.cat((node_ne['host'], node_mp['host']), dim=1)
        phage_fused = torch.cat((node_ne['phage'], node_mp['phage']), dim=1)

        final_phage = self.dropout(self.attn(phage_fused))
        final_host = self.dropout(self.attn(host_fused))

        train_edges = mask.nonzero(as_tuple=False)
        phage_embs = final_phage[train_edges[:, 0]]
        host_embs = final_host[train_edges[:, 1]]
        
        logits = self.decoder(torch.cat([phage_embs, host_embs], dim=1)).squeeze()
        loss = nn.BCEWithLogitsLoss()(logits, phage_host[mask.bool()].float())
        loss += self.contrast(node_mp, node_ne, pos_dict) * self.cl_weight

        with torch.no_grad():
            num_phages, num_hosts = final_phage.shape[0], final_host.shape[0]
            phage_embs_all = final_phage.unsqueeze(1).expand(-1, num_hosts, -1)
            host_embs_all = final_host.unsqueeze(0).expand(num_phages, -1, -1)
            logits_all = self.decoder(torch.cat([phage_embs_all, host_embs_all], dim=2).view(-1, self.dim_embedding*2))
            predict_all = torch.sigmoid(logits_all).view(num_phages, num_hosts)
            
        return predict_all, loss