"""集中读取配置. 所有路径相对仓库根目录, 不写死任何机器上的绝对路径."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))  # 可在 .env 里设 DATA_DIR, 把数据目录放到仓库外
INDEX_DIR = DATA_DIR / "index"

# 大模型调用 (rag/llm.py, OpenAI 兼容接口); 检索和检索评测不用这几项
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")  # "openai" = OpenAI 兼容接口 (DeepSeek 经 OpenRouter); "anthropic" = Claude 原生接口 (rag/claude_llm.py, key 读 ANTHROPIC_API_KEY)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "")  # 本项目用 low; 留空时不传
# 经 OpenRouter 时只用这几家服务商, 按顺序优先 (逗号分隔, 名字同 OpenRouter 模型页的 provider 列表); 由 rag/openrouter.py
# 加进请求. 默认两家都是 fp8, 支持 tools; InferenceNet 更便宜, GMICloud 更快 (选的时候的观察, 价格和速度会变).
# 设成空字符串 (LLM_PROVIDERS=) 时不限定, 由 OpenRouter 自己选.
LLM_PROVIDERS = [x.strip() for x in os.getenv("LLM_PROVIDERS", "InferenceNet,GMICloud").split(",") if x.strip()]

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
