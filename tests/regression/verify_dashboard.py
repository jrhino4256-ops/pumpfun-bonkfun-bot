"""Verify the web dashboard guards its API, edits configs safely and runs bots.

The dashboard (`src/dashboard/`) can start a bot that spends from the wallet,
so every API call needs the startup token, and a bot id can only name a file
in bots/. Settings are written line by line so the YAML keeps its comments;
only whitelisted keys are writable and the result must pass the bot's own
config validation, or nothing is written.

Offline machine checks against a temporary copy of bots/; no network, no
funds moved, and the "bot" it starts is a stand-in process:

  1. /api calls without the token, or with a wrong one, get 401.
  2. A bot id outside bots/*.yaml is a 404, including path traversal.
  3. Editing every whitelisted setting on every committed config keeps all
     comments, reloads with the new values and passes `validate_config`;
     commented-out keys are uncommented, missing sections are added.
  4. Unknown keys, wrong types and out-of-range values are refused and the
     file is left untouched.
  5. A started bot shows as running, refuses a settings edit and a second
     start, stops on SIGINT, and its console output is captured.
  6. AI verdict log lines parse; trades pair buy to sell with the right P&L.

Usage:
    uv run tests/regression/verify_dashboard.py
"""

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from config_loader import validate_config  # noqa: E402
from dashboard.server import (  # noqa: E402
    TOKEN_HEADER,
    Dashboard,
    create_app,
    parse_ai_decisions,
    summarize_trades,
)
from dashboard.yaml_edit import (  # noqa: E402
    SettingError,
    apply_settings,
    read_settings,
)

TOKEN = "test-token"  # noqa: S105
AUTH = {TOKEN_HEADER: TOKEN}

# A stand-in bot: prints, then idles until SIGINT, then exits 0.
FAKE_BOT = (
    "import signal, sys, time\n"
    "signal.signal(signal.SIGINT, lambda *_: sys.exit(0))\n"
    "print('fake bot up', flush=True)\n"
    "while True: time.sleep(0.05)\n"
)

EDITS = {
    "enabled": False,
    "trade.buy_amount": 0.002,
    "trade.buy_slippage": 0.2,
    "trade.sell_slippage": 0.35,
    "trade.exit_strategy": "tp_sl",
    "trade.take_profit_percentage": 0.5,
    "trade.stop_loss_percentage": 0.15,
    "trade.max_hold_time": 90,
    "trade.price_check_interval": 1,
    "filters.yolo_mode": True,
    "ai_filter.enabled": True,
    "ai_filter.model": "claude-haiku-5-5",
    "ai_filter.min_score": 80,
    "ai_filter.effort": "medium",
    "ai_filter.fail_open": False,
    "ai_filter.strategy_prompt": 'Prefer "AI" themes # not a comment\nsecond line',
}


def _check(label: str, passed: bool, detail: str = "") -> bool:  # noqa: FBT001
    print(
        f"  [{'PASS' if passed else 'FAIL'}] {label}{': ' + detail if detail else ''}"
    )
    return passed


def _comments(text: str) -> list[str]:
    """Full-line comments, which an edit must never drop."""
    return [ln.strip() for ln in text.splitlines() if ln.strip().startswith("#")]


def _make_root() -> Path:
    root = Path(tempfile.mkdtemp(prefix="dashboard-verify-"))
    shutil.copytree(PROJECT_ROOT / "bots", root / "bots")
    (root / "logs").mkdir()
    (root / "trades").mkdir()
    return root


async def check_auth_and_paths(client: TestClient) -> bool:
    print("\n1-2. Token and bot ids")
    ok = True
    for label, headers in (("no token", {}), ("wrong token", {TOKEN_HEADER: "nope"})):
        resp = await client.get("/api/bots", headers=headers)
        ok &= _check(label, resp.status == 401, str(resp.status))
    resp = await client.get("/api/bots", headers=AUTH)
    ok &= _check("right token", resp.status == 200, str(resp.status))
    resp = await client.get("/")
    ok &= _check("page itself needs no token", resp.status == 200, str(resp.status))
    for bot_id in ("nope", "..%2F..%2Fpyproject", "..%2F.env"):
        resp = await client.post(f"/api/bots/{bot_id}/start", headers=AUTH)
        ok &= _check(f"bot id {bot_id!r}", resp.status == 404, str(resp.status))
    return ok


def check_edits() -> bool:
    print("\n3. Editing every whitelisted setting on every committed config")
    ok = True
    for path in sorted((PROJECT_ROOT / "bots").glob("*.yaml")):
        before = path.read_text()
        try:
            after = apply_settings(before, EDITS)
        except SettingError as e:
            ok &= _check(path.name, False, str(e))
            continue
        data = yaml.safe_load(after)
        validate_config(data)
        values = read_settings(data)
        wrong = {k: values[k] for k, v in EDITS.items() if values[k] != v}
        lost = [c for c in _comments(before) if c not in _comments(after)]
        # The only comment allowed to vanish is a commented-out key now live.
        lost = [
            c
            for c in lost
            if not any(c.lstrip("# ").startswith(k.split(".")[-1] + ":") for k in EDITS)
        ]
        ok &= _check(
            path.name, not wrong and not lost, f"wrong={wrong} lost={lost[:2]}"
        )
        # A second pass over the edited file must be stable (block scalars etc.).
        again = apply_settings(after, {"ai_filter.strategy_prompt": "short"})
        ok &= _check(
            f"{path.name} re-edit",
            read_settings(yaml.safe_load(again))["ai_filter.strategy_prompt"]
            == "short",
        )
    return ok


def check_refusals() -> bool:
    print("\n4. Refused edits")
    text = (PROJECT_ROOT / "bots" / "bot-sniper-1-geyser.yaml").read_text()
    ok = True
    for label, change in (
        ("unknown key", {"private_key": "x"}),
        ("rpc endpoint", {"rpc_endpoint": "https://evil"}),
        ("wrong type", {"ai_filter.min_score": "80"}),
        ("bool as int", {"ai_filter.enabled": 1}),
        ("out of range", {"ai_filter.min_score": 101}),
        ("bad choice", {"ai_filter.model": "gpt"}),
        ("fractional int", {"trade.max_hold_time": 1.5}),
    ):
        try:
            apply_settings(text, change)
            ok &= _check(label, False, "accepted")
        except SettingError as e:
            ok &= _check(label, True, str(e))
    return ok


async def check_process(client: TestClient, dashboard: Dashboard) -> bool:
    print("\n5. Start, refuse edits while running, stop")
    bot_id = "bot-sniper-7-ai"
    resp = await client.post(f"/api/bots/{bot_id}/start", headers=AUTH)
    ok = _check(
        "start", resp.status == 200 and (await resp.json())["running"], str(resp.status)
    )
    resp = await client.post(f"/api/bots/{bot_id}/start", headers=AUTH)
    ok &= _check("second start refused", resp.status == 409, str(resp.status))
    path = dashboard.config_path(bot_id)
    before = path.read_text()
    resp = await client.put(
        f"/api/bots/{bot_id}/settings", headers=AUTH, json={"ai_filter.min_score": 90}
    )
    ok &= _check(
        "edit refused while running",
        resp.status == 409 and path.read_text() == before,
        str(resp.status),
    )
    await asyncio.sleep(0.5)
    resp = await client.get(f"/api/bots/{bot_id}/log", headers=AUTH)
    console = (await resp.json())["console"]
    ok &= _check("console captured", "fake bot up" in console, str(console))
    resp = await client.post(f"/api/bots/{bot_id}/stop", headers=AUTH)
    ok &= _check("stop accepted", resp.status == 200, str(resp.status))
    process = dashboard.processes[bot_id].process
    await asyncio.wait_for(process.wait(), 5)
    ok &= _check(
        "exited cleanly on SIGINT", process.returncode == 0, str(process.returncode)
    )

    resp = await client.put(
        f"/api/bots/{bot_id}/settings", headers=AUTH, json={"ai_filter.min_score": 101}
    )
    ok &= _check(
        "invalid edit is a 400, file untouched",
        resp.status == 400 and path.read_text() == before,
    )
    resp = await client.put(
        f"/api/bots/{bot_id}/settings", headers=AUTH, json={"ai_filter.min_score": 90}
    )
    data = await resp.json()
    ok &= _check(
        "edit when stopped",
        resp.status == 200 and data["settings"]["ai_filter.min_score"] == 90,
    )
    return ok


def check_parsing(root: Path) -> bool:
    print("\n6. AI verdicts and trades")
    lines = [
        "2026-10-09 12:00:01 - trading.universal_trader - INFO - AI approved CAT - score 82: strong theme",
        "2026-10-09 12:00:02 - trading.universal_trader - INFO - Skipping <img src=x> - AI score 12 (min 75): random ticker",
        "2026-10-09 12:00:03 - trading.universal_trader - INFO - Skipping DOG - AI score n/a (min 75): model call exceeded 10s",
    ]
    d = parse_ai_decisions(lines)
    ok = _check(
        "three verdicts, newest first",
        [x["score"] for x in d] == [None, 12, 82],
        str(d),
    )
    ok &= _check(
        "symbol kept verbatim for safe rendering", d[1]["symbol"] == "<img src=x>"
    )

    trades = root / "trades" / "trades.log"
    rows = [
        {
            "action": "buy",
            "token_address": "A",
            "symbol": "A",
            "price": 1.0,
            "amount": 10,
        },
        {
            "action": "buy",
            "token_address": "B",
            "symbol": "B",
            "price": 2.0,
            "amount": 5,
        },
        {
            "action": "sell",
            "token_address": "A",
            "symbol": "A",
            "price": 1.5,
            "amount": 10,
        },
        {
            "action": "buy",
            "token_address": "C",
            "symbol": "C",
            "price": 1.0,
            "amount": 1,
        },
        {
            "action": "sell",
            "token_address": "B",
            "symbol": "B",
            "price": 1.0,
            "amount": 5,
        },
    ]
    trades.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    s = summarize_trades(trades)
    summary = s["summary"]
    ok &= _check(
        "P&L",
        abs(summary["pnl"] - 0.0) < 1e-9
        and summary["wins"] == 1
        and summary["losses"] == 1,
        str(summary),
    )
    ok &= _check("open positions", [o["token_address"] for o in s["open"]] == ["C"])
    return ok


async def main() -> None:
    root = _make_root()
    try:
        dashboard = Dashboard(root, TOKEN)
        dashboard.bot_command = lambda _path: [sys.executable, "-c", FAKE_BOT]
        async with TestClient(TestServer(create_app(dashboard))) as client:
            results = [
                await check_auth_and_paths(client),
                check_edits(),
                check_refusals(),
                await check_process(client, dashboard),
                check_parsing(root),
            ]
    finally:
        shutil.rmtree(root, ignore_errors=True)
    if all(results):
        print("\nAll checks passed.")
    else:
        print("\nSome checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
