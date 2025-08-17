import os
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.absolute()

# 定义文档目录
DOCS_DIR = ROOT_DIR / 'docs'

# 定义案例目录
CASE_DIR = ROOT_DIR / 'case'
CASE_OUTPUT_DIR = CASE_DIR / 'output'


# 确保文档目录存在
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(CASE_DIR, exist_ok=True)
os.makedirs(CASE_OUTPUT_DIR, exist_ok=True)


# 案例文件路径
# DXF_FILE = CASE_DIR / 'input.dxf'
# DXF_FILE = CASE_DIR / 'integrated-wlq.dxf'
DXF_FILE = CASE_DIR / 'AIMIC.dxf'
JSON_FILE = CASE_OUTPUT_DIR / 'building_origin.json'
MATERIAL_FILE = DOCS_DIR / 'FEA_semantic_lists3.json'