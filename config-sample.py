"""
配置文件：定义系统的基本配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# API配置
AIHUBMIX_API_KEY = os.getenv("Aihubmix_API_KEY_ZZT")
SILICONFLOW_API_KEY = os.getenv("Siliconflow_API_KEY")
DOUBAO_API_KEY = os.getenv("Doubao_API_KEY")

# 模型配置
MODEL_CONFIGS = {
    "gpt": {
        "model": "gpt-5.2",
        "api_key": AIHUBMIX_API_KEY,
        "base_url": "https://aihubmix.com/v1",
        "temperature": 0.0,
    },
    "qwen": {
        "model": "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "api_key": SILICONFLOW_API_KEY,
        "base_url": "https://api.siliconflow.cn/v1",
        "temperature": 0.0,
    },
    "doubao": {
        "model": "doubao-seed-1-6-251015",
        "api_key": DOUBAO_API_KEY,
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "temperature": 0.0,
    },
    "claude": {
        "model": "claude-sonnet-4-5",
        "api_key": AIHUBMIX_API_KEY,
        "base_url": "https://aihubmix.com/v1",
        "temperature": 0.0,
    },
    "gemini": {
        "model": "gemini-2.5-pro",
        "api_key": AIHUBMIX_API_KEY,
        "base_url": "https://aihubmix.com/v1",
        "temperature": 0.0,
    },
}

# 默认使用的模型
DEFAULT_MODEL = "gpt"

test_accuracy_model = "gpt"

# 各任务使用的模型配置（如果未指定则使用 DEFAULT_MODEL）
TASK_MODELS = {
    "router": "qwen",      # 路由任务使用的模型（用于分析问题类型）
    "validator": "qwen",    # 验证任务使用的模型（用于验证增强图片）
    "answer": "claude-sonnet-4-5",       # 答案生成任务使用的模型（用于生成最终答案）
}

# 验证配置
MAX_VALIDATION_ATTEMPTS = 1  # 最大验证尝试次数
VALIDATION_RETRY_DELAY = 1.0  # 验证重试延迟（秒）

# 输出目录
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# 临时文件目录
TEMP_DIR = OUTPUT_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# 绘图配置
DRAWING_CONFIG = {
    "point_color": "blue",      # 点的颜色
    "point_marker": "circle",   # 点的标记样式 ('circle', 'square', 'x')
    "point_size": 9,           # 点的大小（像素）
}
