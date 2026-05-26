import fitz
import os

path = r'E:\03-自我学习\copp\reference\Online time-optimal trajectory planning along parametric toolpaths with strict constraint satisfaction and certifiable feasibility guarantee.pdf'
doc = fitz.open(path)
text = ''
for i in range(doc.page_count):
    text += f'\n\n=== Page {i+1} ===\n'
    text += doc[i].get_text()

out = r'E:\03-自我学习\copp\reference\extracted_text.txt'
with open(out, 'w', encoding='utf-8') as f:
    f.write(text)
print(f"Done: {doc.page_count} pages, {len(text)} chars")
