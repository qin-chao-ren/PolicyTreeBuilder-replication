"""
三模型适配器: Anthropic Claude / OpenAI GPT / Google Gemini

每个 client 提供统一接口:
  client.complete(system: str, user: str) -> str
返回模型原始文本（应是 JSON）。

依赖（按需安装）:
  pip install anthropic openai google-generativeai

环境变量:
  ANTHROPIC_API_KEY
  OPENAI_API_KEY
  GOOGLE_API_KEY (或 GEMINI_API_KEY)

模型字符串可在 MODEL_CONFIG 中调整。
"""
import os
import time
import json
from pathlib import Path


# ============================
# 自动加载 .env (项目根目录或当前目录)
# ============================
def _load_dotenv():
    candidates = [
        Path(__file__).resolve().parent.parent / ".env",   # eval/.env
        Path(__file__).resolve().parent / ".env",          # eval/scripts/.env
        Path.cwd() / ".env",
    ]
    for p in candidates:
        if not p.exists():
            continue
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            # 去掉可能存在的 export 前缀
            if k.startswith("export "):
                k = k[len("export "):].strip()
            # 去引号
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            # 不覆盖已经显式设置的环境变量
            if k and k not in os.environ:
                os.environ[k] = v
        break  # 只加载找到的第一个 .env

_load_dotenv()


# ============================
# 模型字符串 - 启动前确认/修改
# ============================
# 每个 judge 三个关键字段:
#   model         - 该 endpoint 上的模型名（请按各家文档填）
#   base_url      - OpenAI 兼容 endpoint 的根地址（不要带尾斜杠）
#   api_key_env   - 从哪个环境变量读 key
#   timeout       - 单次HTTP请求超时秒数 (可选, 默认60)
# 下面三组占位你按实际改就行；key 名建议各家分开，不要复用 OPENAI_API_KEY。
MODEL_CONFIG = {
    # These public defaults preserve the judge keys used by the archived
    # evaluation outputs. Replace model/base-url environment variables locally
    # before running new API calls.
    "A_kimi": {
        "provider": "openai_compat",
        "model": os.environ.get("A_KIMI_MODEL", "kimi-k2.5"),
        "base_url_env": "A_KIMI_BASE_URL",
        "api_key_env": "A_KIMI_API_KEY",
        "temperature": 0.0,
        "max_tokens": 1024,
        "json_mode": True,
    },
    "B_claude": {
        "provider": "openai_compat",
        "model": os.environ.get("B_CLAUDE_MODEL", "claude-opus-4-6"),
        "base_url_env": "B_CLAUDE_BASE_URL",
        "api_key_env": "B_CLAUDE_API_KEY",
        "temperature": 0.0,
        "max_tokens": 1024,
        "json_mode": True,
        "timeout": 60,
    },
    "C_gemini": {
        "provider": "openai_compat",
        "model": os.environ.get("C_GEMINI_MODEL", "gemini-pro"),
        "base_url_env": "C_GEMINI_BASE_URL",
        "api_key_env": "C_GEMINI_API_KEY",
        "temperature": 0.0,
        "max_tokens": 16384,
        "json_mode": True,
        "timeout": 90,
    },
}


# ============================
# 通用接口
# ============================
class BaseClient:
    name: str
    model: str
    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError
 
 
# ============================
# Anthropic
# ============================
class AnthropicClient(BaseClient):
    def __init__(self, model, temperature=0.0, max_tokens=1024):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.name = "anthropic"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
 
    def complete(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # text content blocks
        out = []
        for blk in resp.content:
            if getattr(blk, "type", None) == "text":
                out.append(blk.text)
        return "".join(out)
 
 
# ============================
# OpenAI-compatible (任何走 OpenAI SDK 协议的 endpoint)
# ============================
class OpenAICompatClient(BaseClient):
    def __init__(self, model, base_url, api_key_env,
                 temperature=0.0, max_tokens=1024, json_mode=True,
                 timeout=60.0):
        from openai import OpenAI
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"环境变量 {api_key_env} 未设置")
        # timeout: 单次HTTP调用最长等待秒数 (默认600s太长会导致挂死)
        # max_retries=0: 关闭SDK自带重试, 由我们的call_with_retry控制
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )
        self.name = "openai_compat"
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.json_mode = json_mode
 
    def complete(self, system: str, user: str) -> str:
        kwargs = dict(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
 
        # 防御: 某些不兼容 endpoint 会返回字符串/dict, 而不是 ChatCompletion 对象
        if isinstance(resp, str):
            raise RuntimeError(
                f"endpoint {self.base_url} 返回了字符串而非ChatCompletion对象, "
                f"前200字: {resp[:200]!r}. 该endpoint可能不是OpenAI兼容协议."
            )
        if not hasattr(resp, "choices"):
            raise RuntimeError(
                f"endpoint {self.base_url} 返回对象无.choices字段, type={type(resp).__name__}, "
                f"repr前200字: {repr(resp)[:200]}"
            )
        return resp.choices[0].message.content or ""
 
 
# ============================
# Google Gemini
# ============================
class GeminiClient(BaseClient):
    def __init__(self, model, temperature=0.0, max_tokens=1024):
        import google.generativeai as genai
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY required")
        genai.configure(api_key=api_key)
        self.genai = genai
        self.name = "google"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
 
    def complete(self, system: str, user: str) -> str:
        m = self.genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "response_mime_type": "application/json",
            },
        )
        resp = m.generate_content(user)
        return resp.text or ""
 
 
# ============================
# 工厂
# ============================
def make_client(judge_key: str) -> BaseClient:
    cfg = MODEL_CONFIG[judge_key]
    p = cfg["provider"]
    if p == "openai_compat":
        base_url = cfg.get("base_url") or os.environ.get(cfg.get("base_url_env", ""))
        if not base_url:
            raise RuntimeError(
                f"Base URL is not configured for judge {judge_key}. "
                f"Set {cfg.get('base_url_env')} or edit MODEL_CONFIG."
            )
        return OpenAICompatClient(
            model=cfg["model"],
            base_url=base_url,
            api_key_env=cfg["api_key_env"],
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 1024),
            json_mode=cfg.get("json_mode", True),
            timeout=cfg.get("timeout", 60.0),
        )
    if p == "anthropic":
        return AnthropicClient(cfg["model"], cfg["temperature"], cfg["max_tokens"])
    if p == "google":
        return GeminiClient(cfg["model"], cfg["temperature"], cfg["max_tokens"])
    raise ValueError(f"unknown provider: {p}")
 
 
# ============================
# JSON 安全解析
# ============================
def _extract_balanced_json(text: str):
    """
    从文本中提取第一个出现的、括号配平的 JSON 对象。
    用于处理模型在 JSON 前后混入 reasoning / 解释文字的情况。
    返回 (json_str, start_idx, end_idx) 或 None。
    """
    n = len(text)
    i = 0
    while i < n:
        if text[i] == "{":
            depth = 0
            in_str = False
            escape = False
            for j in range(i, n):
                ch = text[j]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return text[i:j+1], i, j+1
            # 没配平就跳出，去试下一个 {
            break
        i += 1
    return None
 
 
def _try_fix_unescaped_inner_quotes(s: str) -> str:
    """
    Gemini 等模型常见 bug: 在 JSON 字符串值里嵌入未转义的双引号。
    例: "issue_summary": "包含"未转义引号"的内容"
    扫描整个字符串, 用状态机识别字符串边界, 把疑似在字符串内部的裸双引号替换为 \\"。
 
    判定规则: 一个 " 是真正的字符串边界, 当且仅当它紧邻的下一个非空白字符
    是  ,  :  }  ]  或者文本末尾。否则视为字符串内部的裸引号, 转义之。
    """
    out = []
    n = len(s)
    in_string = False
    i = 0
    while i < n:
        ch = s[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue
 
        # 在字符串内
        if ch == "\\":
            # 保留转义对
            out.append(ch)
            if i + 1 < n:
                out.append(s[i + 1])
                i += 2
            else:
                i += 1
            continue
 
        if ch == '"':
            # 判断是否为真正的字符串结束
            j = i + 1
            while j < n and s[j] in " \t\r\n":
                j += 1
            next_meaningful = s[j] if j < n else ""
            if next_meaningful in (",", ":", "}", "]", "") or j >= n:
                out.append(ch)
                in_string = False
                i += 1
            else:
                # 字符串内部的裸引号, 转义之
                out.append("\\")
                out.append('"')
                i += 1
            continue
 
        out.append(ch)
        i += 1
 
    return "".join(out)
 
 
def parse_json_safe(text: str):
    """容忍 ```json fences、前后噪声文字（如 thinking）、reasoning 模型混合输出、
    以及字符串值内未转义的双引号(Gemini 常见 bug)。"""
    t = (text or "").strip()
    if not t:
        raise ValueError("empty response text")
 
    # 1) 去 ```json ... ``` fence
    if t.startswith("```"):
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1:]
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
 
    # 2) 直接尝试解析
    try:
        return json.loads(t)
    except Exception:
        pass
 
    # 3) 提取首个配平的 {...} 对象
    found = _extract_balanced_json(t)
    if found:
        candidate, _, _ = found
        try:
            return json.loads(candidate)
        except Exception:
            pass
 
    # 4) 修复字符串内裸引号后再试
    fixed = _try_fix_unescaped_inner_quotes(t)
    if fixed != t:
        try:
            return json.loads(fixed)
        except Exception:
            pass
        found2 = _extract_balanced_json(fixed)
        if found2:
            try:
                return json.loads(found2[0])
            except Exception:
                pass
        # 兜底: fixed 上做花括号截取
        if "{" in fixed and "}" in fixed:
            fi = fixed.find("{"); fj = fixed.rfind("}")
            if fj > fi:
                try:
                    return json.loads(fixed[fi:fj+1])
                except Exception:
                    pass
 
    # 5) 兜底: 简单首尾花括号截取（旧逻辑）
    if "{" in t and "}" in t:
        i = t.find("{"); j = t.rfind("}")
        if j > i:
            try:
                return json.loads(t[i:j+1])
            except Exception:
                pass
 
    raise ValueError(f"无法从返回中解析出JSON. 前300字: {t[:300]!r}")
 
 
# ============================
# 重试包装
# ============================
def call_with_retry(client: BaseClient, system: str, user: str,
                    max_retries: int = 3, base_delay: float = 2.0):
    last_err = None
    last_text = ""
    for attempt in range(max_retries):
        try:
            text = client.complete(system, user)
            last_text = text
            obj = parse_json_safe(text)
            return obj, text
        except KeyboardInterrupt:
            # 不吞 Ctrl-C, 让用户能立即终止
            raise
        except Exception as e:
            last_err = e
            # 截短错误信息: HTML/超长返回会刷屏
            err_repr = f"{type(e).__name__}: {e}"
            if len(err_repr) > 200:
                err_repr = err_repr[:200] + "...(truncated)"
            wait = base_delay * (2 ** attempt)
            if attempt + 1 < max_retries:
                print(f"  [retry {attempt+1}/{max_retries}] {err_repr}; sleep {wait:.1f}s")
                # 分段 sleep, 让 Ctrl-C 能立即响应
                _interruptible_sleep(wait)
            else:
                print(f"  [retry {attempt+1}/{max_retries}] {err_repr}; giving up")
    raise RuntimeError(f"all retries failed: {last_err}|||LAST_RAW|||{last_text}")
 
 
def _interruptible_sleep(seconds: float):
    """time.sleep 的包装, 默认就支持 Ctrl-C, 这里只是显式化"""
    end = time.time() + seconds
    while True:
        remain = end - time.time()
        if remain <= 0:
            break
        time.sleep(min(remain, 0.5))
