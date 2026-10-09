"""Edit whitelisted bot YAML settings in place, keeping the file's comments.

A YAML round trip through `yaml.safe_dump` would drop every comment in
`bots/*.yaml`, and those comments are the configs' documentation, so values are
rewritten line by line instead. Only the keys in `SETTINGS` can be written.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

import yaml

from config_loader import validate_config


@dataclass(frozen=True)
class Setting:
    """One editable key: its type and the values it may take."""

    kind: type
    low: float | None = None
    high: float | None = None
    choices: tuple[str, ...] | None = None
    max_length: int = 2000


SETTINGS: dict[str, Setting] = {
    "enabled": Setting(bool),
    "trade.buy_amount": Setting(float, 0.000001, 1_000),
    "trade.buy_slippage": Setting(float, 0, 1),
    "trade.sell_slippage": Setting(float, 0, 1),
    "trade.exit_strategy": Setting(str, choices=("time_based", "tp_sl", "manual")),
    "trade.take_profit_percentage": Setting(float, 0.001, 100),
    "trade.stop_loss_percentage": Setting(float, 0.001, 1),
    "trade.max_hold_time": Setting(int, 1, 86_400),
    "trade.price_check_interval": Setting(int, 1, 3_600),
    "filters.yolo_mode": Setting(bool),
    "ai_filter.enabled": Setting(bool),
    "ai_filter.model": Setting(
        str, choices=("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")
    ),
    "ai_filter.min_score": Setting(int, 0, 100),
    "ai_filter.effort": Setting(str, choices=("low", "medium", "high", "xhigh", "max")),
    "ai_filter.fail_open": Setting(bool),
    "ai_filter.strategy_prompt": Setting(str),
}

# A quoted scalar, or a bare one running up to an inline comment.
_VALUE = r"(?:\"(?:[^\"\\]|\\.)*\"|'[^']*'|[^#\n]*?)"
_BLOCK_SCALAR = re.compile(r"^[|>][-+0-9]*$")


class SettingError(ValueError):
    """A setting the dashboard refuses to write; the message is user-facing."""


def coerce(path: str, value: Any) -> Any:
    """Check `value` against the whitelist and return it in the setting's type.

    Raises:
        SettingError: unknown key, wrong type, or out of range.
    """
    setting = SETTINGS.get(path)
    if setting is None:
        raise SettingError(f"{path} is not an editable setting")

    if setting.kind is bool:
        if not isinstance(value, bool):
            raise SettingError(f"{path} must be true or false")
        return value
    if setting.kind is str:
        if not isinstance(value, str):
            raise SettingError(f"{path} must be text")
        if setting.choices and value not in setting.choices:
            raise SettingError(f"{path} must be one of {', '.join(setting.choices)}")
        if len(value) > setting.max_length:
            raise SettingError(f"{path} is limited to {setting.max_length} characters")
        return value

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SettingError(f"{path} must be a number")
    if setting.kind is int:
        if value != int(value):
            raise SettingError(f"{path} must be a whole number")
        value = int(value)
    else:
        value = float(value)
    if not setting.low <= value <= setting.high:
        raise SettingError(f"{path} must be between {setting.low} and {setting.high}")
    return value


def read_settings(data: dict) -> dict[str, Any]:
    """The whitelisted settings present in a loaded config; absent keys are None."""
    values = {}
    for path in SETTINGS:
        node: Any = data
        for key in path.split("."):
            node = node.get(key) if isinstance(node, dict) else None
        values[path] = node
    return values


def apply_settings(text: str, changes: dict[str, Any]) -> str:
    """Return `text` with each change written, after validating the result.

    Raises:
        SettingError: a change is refused, or the edited config fails the
            bot's own validation.
    """
    coerced = {path: coerce(path, value) for path, value in changes.items()}
    for path, value in coerced.items():
        text = _set_value(text, path, value)

    try:
        data = yaml.safe_load(text)
        validate_config(data)
    except (yaml.YAMLError, ValueError) as e:
        raise SettingError(f"edited config is invalid: {e}") from e
    written = read_settings(data)
    for path, value in coerced.items():
        if written[path] != value:
            raise SettingError(f"could not write {path}")
    return text


def _render(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        # A JSON string is a valid YAML double-quoted scalar.
        return json.dumps(value, ensure_ascii=False)
    return repr(value)


def _set_value(text: str, path: str, value: Any) -> str:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    *sections, key = path.split(".")
    rendered = _render(value)

    if not sections:
        start, end, indent = 0, len(lines), ""
        key_indent = r"()"
    else:
        header = re.compile(rf"^{re.escape(sections[0])}:\s*(#.*)?$")
        start = next(
            (i for i, ln in enumerate(lines) if header.match(ln.rstrip("\n"))), None
        )
        if start is None:
            lines.append(f"\n{sections[0]}:\n  {key}: {rendered}\n")
            return "".join(lines)
        start += 1
        end = next(
            (i for i in range(start, len(lines)) if re.match(r"^[^\s#]", lines[i])),
            len(lines),
        )
        indent = next(
            (
                re.match(r"^(\s+)", ln).group(1)
                for ln in lines[start:end]
                if re.match(r"^\s+[^\s#]", ln)
            ),
            "  ",
        )
        key_indent = r"(\s+)"

    pattern = re.compile(
        rf"^{key_indent}(#\s*)?{re.escape(key)}:[ \t]*({_VALUE})([ \t]+#.*)?$"
    )
    matches = [
        (i, m) for i in range(start, end) if (m := pattern.match(lines[i].rstrip("\n")))
    ]
    # Prefer the live key over a commented-out example of it.
    live = [(i, m) for i, m in matches if not m.group(2)]
    chosen = (live or matches or [None])[0]
    if chosen is None:
        lines.insert(start, f"{indent}{key}: {rendered}\n")
        return "".join(lines)

    i, m = chosen
    line_indent, old_value, comment = m.group(1), m.group(3).strip(), m.group(4) or ""
    # "0.4 # Stop loss at 40%" would contradict the new number; an options list
    # beside a string value stays true.
    if isinstance(value, (int, float)) and re.search(r"\d", comment):
        comment = ""
    lines[i] = f"{line_indent}{key}: {rendered}{comment}\n"
    if not m.group(2) and _BLOCK_SCALAR.match(old_value):
        # Drop the old block scalar's body: blank or deeper-indented lines.
        j = i + 1
        while j < end and (
            not lines[j].strip()
            or len(lines[j]) - len(lines[j].lstrip()) > len(line_indent)
        ):
            j += 1
        del lines[i + 1 : j]
    return "".join(lines)
