import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
from Bio import SeqIO
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
import json

# 确保目录下有 fcgr_generator.py
try:
    from fcgr_generator import generate_fcgr
except ImportError:
    print("错误: 找不到 fcgr_generator 模块。请确保 fcgr_generator.py 在同一目录下。")
    sys.exit(1)


class PretrainedFeatureExtractor:
    """使用预训练模型提取特征"""
    
    def __init__(self, model_name='resnet18', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        print(f"加载预训练模型: {model_name}")
        print(f"使用设备: {self.device}")
        
        # 加载预训练模型
        if model_name == 'resnet18':
            base_model = models.resnet18(pretrained=True)
            self.embedding_dim = 512
            # 移除最后的分类层
            self.model = nn.Sequential(*list(base_model.children())[:-1])
            
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ 模型加载成功")
        print(f"  嵌入维度: {self.embedding_dim}")
        
        # ImageNet归一化参数
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def fcgr_to_rgb(self, fcgr):
        """
        将单通道FCGR转换为3通道RGB图像
        
        Args:
            fcgr: [H, W] numpy array
        
        Returns:
            [3, H, W] tensor
        """
        # 归一化到[0, 1]
        fcgr_norm = (fcgr - fcgr.min()) / (fcgr.max() - fcgr.min() + 1e-8)
        
        # 复制到3个通道
        rgb = np.stack([fcgr_norm, fcgr_norm, fcgr_norm], axis=0)
        
        return torch.FloatTensor(rgb)
    
    def extract_embedding(self, fcgr):
        """
        从FCGR提取嵌入向量
        
        Args:
            fcgr: [H, W] numpy array
        
        Returns:
            [embedding_dim] numpy array
        """
        # 转换为RGB
        rgb = self.fcgr_to_rgb(fcgr)
        
        # 调整大小到224×224（ImageNet标准）
        rgb_resized = nn.functional.interpolate(
            rgb.unsqueeze(0),  # [1, 3, H, W]
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )
        
        # 归一化
        rgb_normalized = self.normalize(rgb_resized)
        rgb_normalized = rgb_normalized.to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.model(rgb_normalized)
            
            # 展平（如果需要）
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
            
            embedding = features.cpu().numpy().squeeze()
        
        return embedding
    
    def process_fasta(self, fasta_file, output_dir, kmer=4, start_idx=1):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"处理FASTA文件: {fasta_file}")
        print(f"输出目录: {output_dir}")
        print(f"k-mer: {kmer}")
        print(f"{'='*70}\n")
        
        # 读取序列
        print("读取FASTA文件...")
        try:
            records = list(SeqIO.parse(fasta_file, "fasta"))
        except Exception as e:
            print(f"✗ 错误: {e}")
            sys.exit(1)
        
        print(f"找到 {len(records)} 个序列\n")
        
        if len(records) == 0:
            print("✗ 错误: FASTA文件为空")
            sys.exit(1)
        
        # 统计
        stats = {
            'total': len(records),
            'success': 0,
            'failed': 0,
            'failed_list': []
        }
        
        # 新增：用于存储索引信息的字典
        index_data = {}
        
        # 处理序列
        for i, record in enumerate(tqdm(records, desc="生成嵌入")):
            phage_id = start_idx + i
            
            try:
                seq = str(record.seq).upper()
                seq_length = len(seq)
                
                if seq_length < kmer:
                    raise ValueError(f"序列太短: {seq_length} < {kmer}")
                
                # 生成FCGR
                fcgr = generate_fcgr(seq, k=kmer)
                
                # 提取嵌入
                embedding = self.extract_embedding(fcgr)
                
                # 定义文件名
                filename = f"phage_{phage_id}_embedding.npz"
                filepath = output_dir / filename
                
                # 保存NPZ
                np.savez_compressed(
                    filepath,
                    full_embedding=embedding,
                    phage_id=phage_id,
                    phage_name=record.id,
                    phage_description=record.description,
                    sequence_length=seq_length,
                    kmer=kmer,
                    embedding_dim=self.embedding_dim,
                    method=f'pretrained_{self.model_name}'
                )
                
                ncbi_acc = record.id
                
                # 解析噬菌体名称 (从描述中移除ID部分)
                # Biopython的description通常格式为 "ID Description text"
                phage_name_text = record.description
                if phage_name_text.startswith(ncbi_acc):
                    # 移除开头的ID和可能的空格
                    phage_name_text = phage_name_text[len(ncbi_acc):].strip()
                
                # 如果名称为空，回退使用ID
                if not phage_name_text:
                    phage_name_text = ncbi_acc

                # 构建索引条目
                index_data[str(phage_id)] = {
                    "phage_id": str(phage_id),
                    "embedding_file": filename,
                    "噬菌体ID": str(phage_id),
                    "噬菌体名称": phage_name_text,
                    "NCBI号": ncbi_acc
                }
                
                stats['success'] += 1
                
                if stats['success'] % 100 == 0:
                    tqdm.write(f"  已处理 {stats['success']} 个")
                
            except Exception as e:
                stats['failed'] += 1
                stats['failed_list'].append((phage_id, record.id, str(e)))
                
                error_file = output_dir / f"phage_{phage_id}_error.txt"
                with open(error_file, 'w') as f:
                    f.write(f"Error: {e}\n")
        
        # 新增：保存索引JSON文件
        json_path = output_dir / "embedding_index.json"
        print(f"\n正在保存索引文件: {json_path}")
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                # ensure_ascii=False 保证中文正常显示
                json.dump(index_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠ 保存索引文件失败: {e}")

        # 统计输出
        print(f"\n{'='*70}")
        print("处理完成!")
        print(f"{'='*70}")
        print(f"总序列数: {stats['total']}")
        print(f"成功: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
        print(f"失败: {stats['failed']}")
        
        print(f"\n输出目录: {output_dir.absolute()}")
        print(f"{'='*70}\n")
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description='使用ImageNet预训练模型提取FCGR特征',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
   python generate_embeddings_imagenet.py \\
       -i all_phage_gene.fasta \\
       -o embeddings_resnet18 \\
       --model resnet18
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='输入FASTA文件')
    parser.add_argument('-o', '--output', default='embeddings_pretrained',
                        help='输出目录')
    parser.add_argument('--model', choices=['resnet18'],
                        default='resnet18',
                        help='预训练模型（默认: resnet18）')
    parser.add_argument('--kmer', type=int, default=6,
                        help='k-mer长度')
    parser.add_argument('--start-id', type=int, default=1,
                        help='起始ID')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',
                        help='计算设备')
    
    args = parser.parse_args()
    
    # 检查文件
    if not Path(args.input).exists():
        print(f"✗ 错误: 文件不存在: {args.input}")
        sys.exit(1)
    
    # 创建提取器
    extractor = PretrainedFeatureExtractor(
        model_name=args.model,
        device=args.device
    )
    
    # 处理
    stats = extractor.process_fasta(
        fasta_file=args.input,
        output_dir=args.output,
        kmer=args.kmer,
        start_idx=args.start_id
    )
    
    sys.exit(0 if stats['failed'] == 0 else 1)


if __name__ == "__main__":
    main()