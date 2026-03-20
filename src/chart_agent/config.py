from __future__ import annotations

import os
from dataclasses import dataclass


def _load_env() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
        package_dir = os.path.dirname(__file__)
        repo_env = os.path.normpath(os.path.join(package_dir, "..", ".env"))
        load_dotenv(dotenv_path=repo_env, override=False)
    except Exception:
        pass


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model: str
    api_key: str | None
    base_url: str | None
    temperature: float


_load_env()

AIHUBMIX_API_KEY = _get_env("Aihubmix_API_KEY")
SILICONFLOW_API_KEY = _get_env("Siliconflow_API_KEY")
DOUBAO_API_KEY = _get_env("Doubao_API_KEY")

MODEL_CONFIGS: dict[str, ModelConfig] = {
    "gpt": ModelConfig(
        name="gpt",
        model=_get_env("GPT_MODEL", "gpt-5.2") or "gpt-5.2",
        api_key=AIHUBMIX_API_KEY,
        base_url=_get_env("GPT_BASE_URL", "https://aihubmix.com/v1"),
        temperature=_get_env_float("GPT_TEMPERATURE", 0.0),
    ),
    "qwen": ModelConfig(
        name="qwen",
        model=_get_env("QWEN_MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")
        or "Qwen/Qwen3-VL-235B-A22B-Instruct",
        api_key=SILICONFLOW_API_KEY,
        base_url=_get_env("QWEN_BASE_URL", "https://api.siliconflow.cn/v1"),
        temperature=_get_env_float("QWEN_TEMPERATURE", 0.0),
    ),
    "doubao": ModelConfig(
        name="doubao",
        model=_get_env("DOUBAO_MODEL", "doubao-seed-1-6-251015")
        or "doubao-seed-1-6-251015",
        api_key=DOUBAO_API_KEY,
        base_url=_get_env("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        temperature=_get_env_float("DOUBAO_TEMPERATURE", 0.0),
    ),
    "claude": ModelConfig(
        name="claude",
        model=_get_env("CLAUDE_MODEL", "claude-sonnet-4-6")
        or "claude-sonnet-4-5",
        api_key=AIHUBMIX_API_KEY,
        base_url=_get_env("CLAUDE_BASE_URL", "https://aihubmix.com/v1"),
        temperature=_get_env_float("CLAUDE_TEMPERATURE", 0.0),
    ),
    "gemini": ModelConfig(
        name="gemini",
        model=_get_env("GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-pro",
        api_key=AIHUBMIX_API_KEY,
        base_url=_get_env("GEMINI_BASE_URL", "https://aihubmix.com/v1"),
        temperature=_get_env_float("GEMINI_TEMPERATURE", 0.0),
    ),
}

DEFAULT_MODEL = _get_env("DEFAULT_MODEL", "gpt") or "gpt"

TASK_MODELS: dict[str, str] = {
    "splitter": _get_env("SPLITTER_MODEL", _get_env("ROUTER_MODEL", "gpt") or "gpt") or "gpt",
    "planner": _get_env("PLANNER_MODEL", _get_env("ROUTER_MODEL", "gpt") or "gpt") or "gpt",
    "tool_planner": _get_env("TOOL_PLANNER_MODEL", _get_env("ROUTER_MODEL", "gpt") or "gpt") or "gpt",
    "executor": _get_env("EXECUTOR_MODEL", _get_env("ROUTER_MODEL", "gpt") or "gpt") or "gpt",
    "router": _get_env("ROUTER_MODEL", "gpt") or "gpt",
    "validator": _get_env("VALIDATOR_MODEL", "qwen") or "qwen",
    "answer": _get_env("ANSWER_MODEL", "gpt")
    or "gpt",
}
WEB_SELECTABLE_TASKS: tuple[str, ...] = ("splitter", "planner", "executor", "answer", "tool_planner")


def get_model_config(name: str | None = None) -> ModelConfig:
    model_name = name or DEFAULT_MODEL
    if model_name not in MODEL_CONFIGS:
        available = ", ".join(sorted(MODEL_CONFIGS.keys()))
        raise KeyError(f"Unknown model config '{model_name}'. Available: {available}")
    return MODEL_CONFIGS[model_name]


def get_task_model_config(task: str) -> ModelConfig:
    if task not in TASK_MODELS:
        available = ", ".join(sorted(TASK_MODELS.keys()))
        raise KeyError(f"Unknown task '{task}'. Available: {available}")
    model_value = TASK_MODELS[task]
    if model_value in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_value]
    for config in MODEL_CONFIGS.values():
        if config.model == model_value:
            return config
    available = ", ".join(sorted(MODEL_CONFIGS.keys()))
    raise KeyError(f"Unknown model config '{model_value}'. Available: {available}")


def get_task_model_name(task: str) -> str:
    if task not in TASK_MODELS:
        available = ", ".join(sorted(TASK_MODELS.keys()))
        raise KeyError(f"Unknown task '{task}'. Available: {available}")
    return TASK_MODELS[task]


def resolve_task_model_config(task: str, overrides: dict[str, str] | None = None) -> ModelConfig:
    model_value = ""
    if isinstance(overrides, dict):
        model_value = str(overrides.get(task) or "").strip()
    if not model_value:
        return get_task_model_config(task)
    if model_value in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_value]
    for config in MODEL_CONFIGS.values():
        if config.model == model_value:
            return config
    available = ", ".join(sorted(MODEL_CONFIGS.keys()))
    raise KeyError(f"Unknown model config '{model_value}' for task '{task}'. Available: {available}")


def get_web_model_options() -> list[dict[str, str | float | None]]:
    options: list[dict[str, str | float | None]] = []
    for key, config in sorted(MODEL_CONFIGS.items()):
        options.append(
            {
                "key": key,
                "name": config.name,
                "model": config.model,
                "base_url": config.base_url,
                "temperature": config.temperature,
            }
        )
    return options


def get_web_task_model_defaults() -> dict[str, dict[str, str | None]]:
    defaults: dict[str, dict[str, str | None]] = {}
    for task in WEB_SELECTABLE_TASKS:
        config = get_task_model_config(task)
        defaults[task] = {
            "task": task,
            "selected": get_task_model_name(task),
            "resolved_model": config.model,
            "base_url": config.base_url,
        }
    return defaults


def get_perception_max_retries() -> int:
    return _get_env_int("PERCEPTION_MAX_RETRIES", 2)


def _normalize_svg_perception_mode(mode: str | None) -> str:
    normalized = str(mode or "rules").strip().lower()
    if normalized == "llm":
        return "llm_summary"
    if normalized in {"rules", "llm_summary"}:
        return normalized
    return "rules"


def _normalize_svg_update_mode(mode: str | None) -> str:
    normalized = str(mode or "rules").strip().lower()
    if normalized == "llm":
        return "llm_intent"
    if normalized in {"rules", "llm_intent"}:
        return normalized
    return "rules"


def get_svg_perception_mode(mode: str | None = None) -> str:
    if mode is not None:
        return _normalize_svg_perception_mode(mode)
    mode = _get_env("SVG_PERCEPTION_MODE", "rules")
    return _normalize_svg_perception_mode(mode)


def get_svg_update_mode(mode: str | None = None) -> str:
    if mode is not None:
        return _normalize_svg_update_mode(mode)
    mode = _get_env("SVG_UPDATE_MODE", "rules")
    return _normalize_svg_update_mode(mode)
