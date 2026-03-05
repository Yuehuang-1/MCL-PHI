from Bio import SeqIO
from Bio.Seq import UndefinedSequenceError
import io
import re

def final_robust_extract(input_file, output_file):
    success_count = 0
    error_count = 0
    error_ids = []
    
    # 预编译正则，用于安全提取 ID (匹配 LOCUS 后的第一个单词)
    locus_pattern = re.compile(r'^LOCUS\s+([^\s]+)', re.MULTILINE)

    with open(output_file, "w") as out_f:
        with open(input_file, "r") as in_f:
            content = ""
            for line in in_f:
                content += line
                if line.strip() == "//":
                    # 发现条目结束符，开始尝试解析
                    current_id = "Unknown"
                    try:
                        # 1. 尝试使用正则提取 ID，用于日志记录
                        match = locus_pattern.search(content)
                        if match:
                            current_id = match.group(1)
                        
                        # 2. 将内容转为内存文件
                        handle = io.StringIO(content)
                        record = SeqIO.read(handle, "genbank")
                        
                        # 3. 核心检查：尝试访问序列内容
                        # 这一步会触发 UndefinedSequenceError 如果序列不存在
                        seq_str = str(record.seq) 
                        
                        if len(seq_str) > 0:
                            SeqIO.write(record, out_f, "fasta")
                            success_count += 1
                        else:
                            print(f"跳过空序列条目: {current_id}")
                            error_count += 1
                            
                    except (UndefinedSequenceError, Exception) as e:
                        print(f"无法解析条目 [{current_id}]: {str(e)}")
                        error_count += 1
                        error_ids.append(current_id)
                    
                    # 释放内存，重置内容
                    content = ""

    print("\n" + "="*30)
    print(f"处理完成！统计结果：")
    print(f"✓ 成功提取: {success_count}")
    print(f"✗ 失败跳过: {error_count}")
    if error_ids:
        print(f"失败 ID 示例: {error_ids[:10]}...")
    print("="*30)

# 运行
input_filename = 'sequence.gb' 
output_filename = 'all_phages_genome.fasta'
final_robust_extract(input_filename, output_filename)