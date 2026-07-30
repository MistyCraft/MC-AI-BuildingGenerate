from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SCHEM_DIR = BASE_DIR / "Schamtic_files"
OUTPUT_DIR = BASE_DIR / "Output_files"
TRAINING_DIR = BASE_DIR / "Training_dataset"

# 默认输入和输出结构文件。
DEFAULT_INPUT_SCHEM = SCHEM_DIR / "Ach1.schem"
DEFAULT_OUTPUT_SCHEM = SCHEM_DIR / "items.schem"

# 结构解析过程使用的中间文本文件。
BLOCK_DATA_FILE = OUTPUT_DIR / "block_data.txt"
METADATA_FILE = OUTPUT_DIR / "metadata.txt"
PARSED_BLOCK_DATA_FILE = OUTPUT_DIR / "parsed_block_data.txt"
RAW_BLOCK_DATA_DEBUG_FILE = OUTPUT_DIR / "raw_block_data_debug.txt"
PRINT_BD_FILE = RAW_BLOCK_DATA_DEBUG_FILE
CHECK_TXT2NPY_FILE = OUTPUT_DIR / "check_txt2npy.txt"
PREVIEW_IMAGE_FILE = OUTPUT_DIR / "structure_preview.png"

# 模型训练和数据增强使用的 NPY 文件。
STRUCTURE_NPY_FILE = TRAINING_DIR / "structure.npy"
AUGMENTED_NPY_PREFIX = TRAINING_DIR / "block_data"


def ensure_project_dirs():
    """创建项目运行所需的数据目录。"""
    for directory in (SCHEM_DIR, OUTPUT_DIR, TRAINING_DIR):
        directory.mkdir(parents=True, exist_ok=True)
