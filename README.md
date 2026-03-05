# MCL-PHI
MCL-PHI: A Multi-modal Framework for Phage-Host Interaction Prediction Using Multi-view Contrastive Learning.

本项目通过多模态对比学习框架（MCL-PHI）预测噬菌体与宿主之间的相互作用。项目结合了基因组序列特征（Sequence Features）和文本语义特征（Text Features），利用图神经网络（GNN）和对比学习技术提升预测性能。
## 使用指南
### 第一步：数据准备
1.准备名称列表：使用data/p_name.csv（噬菌体名称/ID）和 h_name.csv（宿主名称/ID）。

2.下载原始数据：根据列表中的 Accession Number，从 NCBI 批量下载基因组数据，保存为 sequence.gb（GenBank格式）。

3.提取序列与生成文本：

运行提取脚本，将 .gb 文件转换为 .fasta 序列文件：
```python batch_download_phages.py```

运行 LLM 脚本，生成文本描述报告：
```python lcel_chain.py```

### 第二步：特征生成
1.生成序列特征:
```python generate_embeddings_imagenet.py -i all_phages_genome.fasta -o embeddings_resnet18```

2.生成文本特征:
```python biobert-v1.1.py```

3.计算相似度：
```python compute_similarity_matrix_simplified.py```
```python compute_similarity.py```

### 第三步：模型训练与评估
使用生成的特征文件运行主程序。
```
  python main.py \
  --seq_feat_path ./embeddings_resnet18_phage \
  --text_feat_path ./phage_embeddings \
  --host_seq_feat_path ./embeddings_resnet18_host \
  --host_text_feat_path ./host_embeddings
```

## 环境依赖
```Python 3.8
torch==1.13.1
transformers==2.1.0
biopython==1.81
numpy==1.21.2
pandas==1.3.4
scikit-learn==1.0.2
scipy==1.6.2
```
