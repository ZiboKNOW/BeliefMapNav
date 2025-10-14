import re


def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

file_path = 'path to your merge_result.txt'  
data = load_text_file(file_path)


spl_values = [float(x) for x in re.findall(r"'spl': ([0-9\.]+)", data)]
no_zeros = [x for x in spl_values if x > 0]
print(f"Number of non-zero SPL values: {len(no_zeros)}")
print(f"sum of non-zero SPL values: {sum(no_zeros)}")

spl_average = sum(spl_values) / len(spl_values) if spl_values else 0
print("length: ",len(spl_values))
print(f"Average SPL: {spl_average}")
print(f"SUM SPL: {sum(spl_values)}")