import re

text_bold = """
**Reasoning:**
...
**Answer:** no
"""

text_plain = """
**Reasoning:**
...
Answer: no
"""

# 使用正则表达式匹配
match = re.search(r'(?:\*\*Answer:\*\*|Answer:)\s*(yes|no)', text_bold, re.IGNORECASE)
if match:
    print(f"匹配结果1: {match.group(1)}") # 输出: 匹配结果1: yes

match = re.search(r'(?:\*\*Answer:\*\*|Answer:)\s*(yes|no)', text_plain, re.IGNORECASE)
if match:
    print(f"匹配结果2: {match.group(1)}") # 输出: 匹配结果2: no