"""Local web dashboard: start and stop bots, edit their settings, and watch
AI decisions, logs and trades.

Binds to 127.0.0.1 by default and requires the token printed at startup on
every API call, because the dashboard can start a bot that spends from the
wallet. A custom header carries the token, so a page on another site cannot
forge a call with a plain form post.
"""

import argparse
import asyncio
import contextlib
import hmac
import ipaddress
import json
import re
import secrets
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from aiohttp import web

from dashboard.yaml_edit import SettingError, apply_settings, read_settings
from utils.logger import get_logger, install_secret_redaction

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
TOKEN_HEADER = "X-Dashboard-Token"

#: Seconds a stopping bot gets to sell and clean up after SIGINT before it is
#: killed. Covers the trader's own 120s shutdown-cleanup budget.
STOP_GRACE_SECONDS = 150.0

_LOG_TAIL_BYTES = 256 * 1024
_LOG_LINE = re.compile(r"^(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
_AI_APPROVED = re.compile(
    r"AI approved (?P<symbol>.+?) - score (?P<score>\d+): (?P<reason>.*)$"
)
_AI_SKIPPED = re.compile(
    r"Skipping (?P<symbol>.+?) - AI score (?P<score>\S+) \(min (?P<min>\d+)\): (?P<reason>.*)$"
)


@dataclass
class BotProcess:
    """A bot the dashboard started."""

    process: asyncio.subprocess.Process
    started_at: float
    console: deque[str] = field(default_factory=lambda: deque(maxlen=200))
    stopping: bool = False


class Dashboard:
    """State behind the HTTP handlers: the bots directory and running processes."""

    def __init__(self, root: Path, token: str) -> None:
        self.root = root
        self.token = token
        self.processes: dict[str, BotProcess] = {}

    def bot_command(self, config_path: Path) -> list[str]:
        """The command that runs one bot; replaced in the offline verifier."""
        return [
            sys.executable,
            str(self.root / "src" / "bot_runner.py"),
            str(config_path),
        ]

    def config_path(self, bot_id: str) -> Path:
        """The YAML for `bot_id` (a file stem), never a path outside bots/.

        Raises:
            web.HTTPNotFound: no such bot.
        """
        candidates = {p.stem: p for p in (self.root / "bots").glob("*.yaml")}
        if bot_id not in candidates:
            raise web.HTTPNotFound(text=json.dumps({"error": "no such bot"}))
        return candidates[bot_id]

    def is_running(self, bot_id: str) -> bool:
        bot = self.processes.get(bot_id)
        return bot is not None and bot.process.returncode is None

    def describe(self, path: Path) -> dict[str, Any]:
        """One bot's card: identity, run state and whitelisted settings."""
        try:
            data = yaml.safe_load(path.read_text()) or {}
            error = None
        except (OSError, yaml.YAMLError) as e:
            data, error = {}, str(e)
        bot = self.processes.get(path.stem)
        running = self.is_running(path.stem)
        return {
            "id": path.stem,
            "name": data.get("name", path.stem),
            "platform": data.get("platform", "pump_fun"),
            "listener": (data.get("filters") or {}).get("listener_type"),
            "running": running,
            "stopping": bool(bot and bot.stopping and running),
            "pid": bot.process.pid if running else None,
            "started_at": bot.started_at if running else None,
            # A stop the user asked for is not a failure, whatever the code.
            "exit_code": (
                None
                if bot is None or running or bot.stopping
                else bot.process.returncode
            ),
            "settings": read_settings(data),
            "error": error,
        }

    async def start(self, bot_id: str) -> None:
        if self.is_running(bot_id):
            raise web.HTTPConflict(text=json.dumps({"error": "already running"}))
        path = self.config_path(bot_id)
        process = await asyncio.create_subprocess_exec(
            *self.bot_command(path),
            cwd=self.root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        bot = BotProcess(process=process, started_at=time.time())
        self.processes[bot_id] = bot
        asyncio.get_running_loop().create_task(self._pump_console(bot))
        logger.info(f"Started bot {bot_id} (pid {process.pid})")

    async def _pump_console(self, bot: BotProcess) -> None:
        assert bot.process.stdout is not None  # noqa: S101
        async for raw in bot.process.stdout:
            bot.console.append(raw.decode(errors="replace").rstrip())

    async def stop(self, bot_id: str) -> None:
        """Ask a bot to shut down cleanly; kill it if it overruns the grace period."""
        bot = self.processes.get(bot_id)
        if bot is None or bot.process.returncode is not None:
            raise web.HTTPConflict(text=json.dumps({"error": "not running"}))
        if bot.stopping:
            return
        bot.stopping = True
        if sys.platform == "win32":
            bot.process.terminate()
        else:
            bot.process.send_signal(signal.SIGINT)
        asyncio.get_running_loop().create_task(self._reap(bot_id, bot))

    async def _reap(self, bot_id: str, bot: BotProcess) -> None:
        try:
            await asyncio.wait_for(bot.process.wait(), STOP_GRACE_SECONDS)
        except TimeoutError:
            logger.warning(
                f"Bot {bot_id} did not stop in {STOP_GRACE_SECONDS}s; killing"
            )
            bot.process.kill()
            await bot.process.wait()
        logger.info(f"Bot {bot_id} exited with {bot.process.returncode}")

    async def shutdown(self) -> None:
        """Stop every running bot, waiting for each to clean up."""
        running = [b for b in self.processes if self.is_running(b)]
        for bot_id in running:
            with contextlib.suppress(web.HTTPException):
                await self.stop(bot_id)
        await asyncio.gather(
            *(self._reap(b, self.processes[b]) for b in running), return_exceptions=True
        )

    def latest_log(self, bot_id: str) -> Path | None:
        """The bot's newest `logs/{name}_*.log`, or None if it never logged."""
        path = self.config_path(bot_id)
        try:
            name = (yaml.safe_load(path.read_text()) or {}).get("name", bot_id)
        except (OSError, yaml.YAMLError):
            name = bot_id
        logs = [
            p
            for p in (self.root / "logs").glob(f"{name}_*.log")
            if re.fullmatch(rf"{re.escape(name)}_\d{{8}}_\d{{6}}\.log", p.name)
        ]
        return max(logs, key=lambda p: p.stat().st_mtime, default=None)


def tail_lines(path: Path, limit: int) -> list[str]:
    """The last `limit` lines of a file, reading at most its final 256 KiB."""
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - _LOG_TAIL_BYTES))
        data = f.read().decode(errors="replace")
    lines = data.splitlines()
    if size > _LOG_TAIL_BYTES:
        lines = lines[1:]  # first line is likely cut mid-way
    return lines[-limit:]


def parse_ai_decisions(lines: list[str]) -> list[dict[str, Any]]:
    """AI verdicts found in bot log lines, newest first."""
    decisions = []
    for line in lines:
        ts = _LOG_LINE.match(line)
        if m := _AI_APPROVED.search(line):
            decisions.append({"buy": True, "score": int(m["score"]), **_common(m, ts)})
        elif m := _AI_SKIPPED.search(line):
            score = int(m["score"]) if m["score"].isdigit() else None
            decisions.append({"buy": False, "score": score, **_common(m, ts)})
    return decisions[::-1]


def _common(m: re.Match, ts: re.Match | None) -> dict[str, Any]:
    return {
        "symbol": m["symbol"],
        "reason": m["reason"],
        "time": ts["ts"] if ts else None,
    }


def summarize_trades(path: Path) -> dict[str, Any]:
    """Trades from `trades/trades.log` paired buy-to-sell per coin.

    P&L is price x amount in the coin's quote asset, before network and
    priority fees, so it overstates what reached the wallet.
    """
    trades = []
    if path.exists():
        for line in tail_lines(path, 2_000):
            with contextlib.suppress(json.JSONDecodeError):
                entry = json.loads(line)
                if isinstance(entry, dict):
                    trades.append(entry)

    open_buys: dict[str, dict] = {}
    closed = []
    for t in trades:
        mint = t.get("token_address")
        try:
            value = float(t.get("price") or 0) * float(t.get("amount") or 0)
        except (TypeError, ValueError):
            continue
        if t.get("action") == "buy":
            open_buys[mint] = {**t, "value": value}
        elif t.get("action") == "sell" and mint in open_buys:
            buy = open_buys.pop(mint)
            pnl = value - buy["value"]
            closed.append(
                {
                    "symbol": t.get("symbol"),
                    "token_address": mint,
                    "bought_at": buy.get("timestamp"),
                    "sold_at": t.get("timestamp"),
                    "cost": buy["value"],
                    "proceeds": value,
                    "pnl": pnl,
                    "pnl_pct": pnl / buy["value"] * 100 if buy["value"] else None,
                }
            )

    wins = sum(1 for c in closed if c["pnl"] > 0)
    return {
        "trades": trades[::-1][:200],
        "closed": closed[::-1],
        "open": list(open_buys.values()),
        "summary": {
            "closed": len(closed),
            "wins": wins,
            "losses": len(closed) - wins,
            "win_rate": wins / len(closed) * 100 if closed else None,
            "pnl": sum(c["pnl"] for c in closed),
        },
    }


def _json(data: Any, status: int = 200) -> web.Response:
    return web.json_response(data, status=status)


def create_app(dashboard: Dashboard) -> web.Application:
    """The aiohttp app; every /api route requires the dashboard token."""

    @web.middleware
    async def require_token(request: web.Request, handler: Any) -> web.StreamResponse:
        if request.path.startswith("/api/"):
            supplied = request.headers.get(TOKEN_HEADER, "")
            if not hmac.compare_digest(supplied.encode(), dashboard.token.encode()):
                return _json({"error": "missing or wrong dashboard token"}, 401)
        return await handler(request)

    async def index(_request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC_DIR / "index.html")

    async def list_bots(_request: web.Request) -> web.Response:
        paths = sorted((dashboard.root / "bots").glob("*.yaml"))
        return _json({"bots": [dashboard.describe(p) for p in paths]})

    async def start_bot(request: web.Request) -> web.Response:
        await dashboard.start(request.match_info["bot"])
        return _json(
            dashboard.describe(dashboard.config_path(request.match_info["bot"]))
        )

    async def stop_bot(request: web.Request) -> web.Response:
        await dashboard.stop(request.match_info["bot"])
        return _json(
            dashboard.describe(dashboard.config_path(request.match_info["bot"]))
        )

    async def update_settings(request: web.Request) -> web.Response:
        bot_id = request.match_info["bot"]
        path = dashboard.config_path(bot_id)
        if dashboard.is_running(bot_id):
            return _json({"error": "stop the bot before changing its settings"}, 409)
        try:
            changes = await request.json()
            if not isinstance(changes, dict):
                raise SettingError("expected an object of setting: value")
            path.write_text(apply_settings(path.read_text(), changes))
        except (SettingError, json.JSONDecodeError) as e:
            return _json({"error": str(e)}, 400)
        return _json(dashboard.describe(path))

    async def bot_log(request: web.Request) -> web.Response:
        bot_id = request.match_info["bot"]
        log = dashboard.latest_log(bot_id)
        lines = tail_lines(log, 2_000) if log else []
        bot = dashboard.processes.get(bot_id)
        return _json(
            {
                "file": log.name if log else None,
                "lines": lines[-300:],
                "ai_decisions": parse_ai_decisions(lines)[:100],
                "console": list(bot.console) if bot else [],
            }
        )

    async def trades(_request: web.Request) -> web.Response:
        return _json(summarize_trades(dashboard.root / "trades" / "trades.log"))

    async def on_cleanup(_app: web.Application) -> None:
        await dashboard.shutdown()

    app = web.Application(middlewares=[require_token])
    app.router.add_get("/", index)
    app.router.add_get("/api/bots", list_bots)
    app.router.add_post("/api/bots/{bot}/start", start_bot)
    app.router.add_post("/api/bots/{bot}/stop", stop_bot)
    app.router.add_put("/api/bots/{bot}/settings", update_settings)
    app.router.add_get("/api/bots/{bot}/log", bot_log)
    app.router.add_get("/api/trades", trades)
    app.on_cleanup.append(on_cleanup)
    return app


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main() -> None:
    """Run the dashboard: `pump_dashboard [--host H] [--port P]`."""
    parser = argparse.ArgumentParser(description="Web dashboard for the trading bots")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    import logging  # noqa: PLC0415

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    install_secret_redaction()

    if not _is_loopback(args.host):
        logger.warning(
            f"Listening on {args.host}: anyone who can reach it and has the token "
            "can start a bot that spends from your wallet."
        )

    token = secrets.token_urlsafe(24)
    dashboard = Dashboard(PROJECT_ROOT, token)
    shown_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host  # noqa: S104
    # print, not logger: the redaction filter would mask the token in the URL.
    print(
        f"\n  Dashboard: http://{shown_host}:{args.port}/#token={token}\n", flush=True
    )
    web.run_app(create_app(dashboard), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
