import re

# 读取外部 txt 文件
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

file_path = '/home/ubuntu/DATA2/zzb/openspcae/outputs/final_results/merge.txt'  # 替换为你的文件路径
data = load_text_file(file_path)

# 使用正则表达式提取所有 'spl' 数值
spl_values = [float(x) for x in re.findall(r"'spl': ([0-9\.]+)", data)]
no_zeros = [x for x in spl_values if x > 0]
print(f"Number of non-zero SPL values: {len(no_zeros)}")
print(f"sum of non-zero SPL values: {sum(no_zeros)}")
# 计算平均值
spl_average = sum(spl_values) / len(spl_values) if spl_values else 0
print("length: ",len(spl_values))
print(f"Average SPL: {spl_average}")
print(f"SUM SPL: {sum(spl_values)}")