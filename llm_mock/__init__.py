"""LLM Mock —— 本地监听的 LLM 请求录制 / 回放代理。

把 LLM 调用录下来，之后按 **精准 / 模糊 / 位置** 三种策略回放；
未命中可以转发真实 API（hybrid）。对外只说 OpenAI Chat Completions 协议，
客户端改一个 base_url 就能用。
"""

from .proxy import Config, MockService, Resp, make_server, serve
from .store import RecordStore
from .canon import exact_key, fuzzy_key, normalize_messages
from .matcher import Matcher

__version__ = "0.1.0"

__all__ = [
    "Config",
    "MockService",
    "Resp",
    "make_server",
    "serve",
    "RecordStore",
    "Matcher",
    "exact_key",
    "fuzzy_key",
    "normalize_messages",
    "__version__",
]
