import os
import csv
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

# ======================
# 1. 环境初始化
# ======================


# ======================
# 2. 核心组件初始化
# ======================
model = ChatOpenAI(
    model="gpt-4o",
    base_url='https://xiaoai.plus/v1',
    api_key="***",
    temperature=0.2
)

# ======================
# 3. 噬菌体分析系统提示
# ======================
system_prompt = """You are an expert in bacteriophage-host associations. Analyze the provided NCBI accession number to characterize the phage and infer its potential host relationships.
Please follow these steps in your analysis, demonstrating your chain of thought:

1. Identification: Identify the phage type and taxonomic features based on the NCBI number
2. Structural Analysis: Analyze the genomic structural features of the phage
3. Protein Function: Identify key proteins encoded by the phage, especially those related to host recognition
4. Host Prediction: Infer the potential host range based on the above information
5. Applications: Evaluate the potential applications of this phage

Here are a few examples:

Example 1:
NCBI: NC_012663
Thinking process:
- Identification: NC_012663 is the genome of Acinetobacter phage AP22, approximately 36.7kb in length, belonging to the order Caudovirales
- Structural Analysis: Double-stranded DNA genome with 44 predicted genes, GC content of 39.1%
- Protein Function: Contains tail fiber protein ORF12 that binds to flagellar surface protein FliC; contains lysozyme gp19
- Host Prediction: Specifically infects Acinetobacter baumannii
- Applications: Can be used for prevention and treatment of Acinetobacter infections, particularly against drug-resistant strains

Example 2:
NCBI: NC_031034
Thinking process:
- Identification: NC_031034 is a Salmonella phage ST4, belonging to the T4-like phage family
- Structural Analysis: Genome approximately 166kb, encoding 265 predicted proteins, with an extended tail
- Protein Function: Contains tail spike protein gp12 that binds to host LPS; tail fiber proteins recognize OmpC receptor
- Host Prediction: Infects Salmonella species, particularly S. typhimurium and S. enteritidis
- Applications: Can be used for food safety detection and treatment of Salmonella infections

Example 3:
NCBI: NC_000866
Thinking process:
- Identification: NC_000866 is E. coli bacteriophage T4, a well-studied member of the Myoviridae family and Caudovirales order
- Structural Analysis: Large dsDNA genome of 168,903 bp with GC content of 35.3%, encoding over 300 gene products
- Protein Function: Contains gp37 and gp38 tail fibers that recognize OmpC and LPS on host surface; gp5/gp27 baseplate forms cell-puncturing device
- Host Prediction: Primarily infects E. coli strains; also has broader activity against some Shigella species
- Applications: Model system for molecular biology; can be used for phage therapy against pathogenic E. coli; valuable for biotechnology applications

Example 4:
NCBI: NC_007021
Thinking process:
- Identification: NC_007021 is Staphylococcus aureus phage P68, a Gram-positive-infecting phage belonging to the Podoviridae family
- Structural Analysis: Compact genome of 18,227 bp with 19-21 putative ORFs; lacks an integrase, suggesting obligate lytic lifestyle
- Protein Function: Encodes 16L and 15L tail proteins that recognize wall teichoic acids; contains lytic enzyme HydH5 with dual catalytic domains
- Host Prediction: Narrow host range limited to Staphylococcus aureus, including MRSA strains
- Applications: Potential therapeutic agent against S. aureus infections; can be used for biocontrol in food processing environments

Example 5:
NCBI: NC_028983
Thinking process:
- Identification: NC_028983 is Pseudomonas aeruginosa phage vB_PaeM_CRL, a giant phage resembling phiKZ
- Structural Analysis: Very large genome of 342,383 bp with low GC content (29.3%) compared to its host; encodes over 400 proteins
- Protein Function: Contains at least 3 different depolymerases targeting different host receptors; possesses nucleus-like compartment for replication
- Host Prediction: Specialized to infect Pseudomonas aeruginosa, particularly biofilm-forming clinical isolates
- Applications: Biofilm degradation agent; potential use against multidrug-resistant P. aeruginosa infections; source of novel antimicrobial enzymes

Please follow the above format to provide a detailed analysis for the new NCBI accession number. Maintain scientific rigor and clearly indicate any uncertain information.
"""

# 构建噬菌体分析提示模板
phage_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "Please analyze the following bacteriophage:\nPhage name: {phage_name}\nNCBI accession: {accession}")
])

# ======================
# 4. 处理链构建
# ======================
# 使用LCEL管道运算符组合组件
phage_chain = phage_prompt | model | StrOutputParser()

# ======================
# 5. 批量处理函数
# ======================
def process_phage_list(input_file: str, output_dir: str = "phage_reports"):
    """
    批量处理噬菌体列表，分析每个噬菌体并保存结果
    
    Args:
        input_file (str): 包含噬菌体信息的CSV文件路径
        output_dir (str): 输出报告的目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取CSV文件
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        phage_list = list(reader)
    
    total_phages = len(phage_list)
    print(f"找到{total_phages}个噬菌体待处理")
    
    # 处理每个噬菌体
    for index, row in enumerate(phage_list, 1):
        if len(row) < 3:
            print(f"警告: 第{index}行格式无效，跳过。")
            continue
            
        phage_id, phage_name, accession = row[0], row[1], row[2]
        print(f"\n正在处理噬菌体 {index}/{total_phages}:")
        print(f"ID: {phage_id}, 名称: {phage_name}, NCBI号: {accession}")
        
        try:
            # 使用LCEL链分析噬菌体
            analysis_result = phage_chain.invoke({
                "phage_name": phage_name,
                "accession": accession
            })
            
            # 保存分析结果到文件
            output_file = os.path.join(output_dir, f"phage_{phage_id}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"噬菌体ID: {phage_id}\n")
                f.write(f"噬菌体名称: {phage_name}\n")
                f.write(f"NCBI号: {accession}\n\n")
                f.write(f"分析结果:\n{analysis_result}")
            
            print(f"分析结果已保存到 {output_file}")
            
        except Exception as e:
            print(f"处理噬菌体 {phage_id} 时出错: {str(e)}")
            # 创建错误报告
            error_file = os.path.join(output_dir, f"phage_{phage_id}_error.txt")
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"处理噬菌体 {phage_id} 时出错:\n")
                f.write(f"名称: {phage_name}\n")
                f.write(f"NCBI号: {accession}\n")
                f.write(f"错误: {str(e)}\n")

# ======================
# 6. 执行入口
# ======================
if __name__ == "__main__":
    # 设置输入文件路径
    input_file = "../../data/p_name.csv"  # 请确保此文件存在
    
    # 处理所有噬菌体
    process_phage_list(input_file)
    
    print("\n所有噬菌体处理完成！")
