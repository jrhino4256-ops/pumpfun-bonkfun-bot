"""Create a fresh trading wallet for the bot and store its key, never printing it.

Writes the key into the env file as SOLANA_PRIVATE_KEY and saves a backup in
Solana CLI keypair format (a JSON array of 64 bytes, which Solflare and
`solana-keygen` import) under ~/.config/pump-bot/. Prints only the address to
fund. Refuses to run when the env file already holds a key: replacing it would
cut the bot off from a wallet that may still hold funds.

Run it on the machine that will run the bot. Losing both the env file and the
backup loses whatever the wallet holds.

Usage:
    uv run tools/create_wallet.py              # writes .env
    uv run tools/create_wallet.py --env-file bots.env
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

from solders.keypair import Keypair

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKUP_DIR = Path.home() / ".config" / "pump-bot"
_KEY_LINE = re.compile(r"^SOLANA_PRIVATE_KEY=(.*)$", re.MULTILINE)


def _write_private(path: Path, text: str) -> None:
    """Write `text` to a file only the owner can read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    path.chmod(0o600)


def main() -> None:
    """Create the wallet, store it, and print the address to fund."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-file", default=str(PROJECT_ROOT / ".env"))
    args = parser.parse_args()
    env_path = Path(args.env_file)

    if not env_path.exists():
        example = PROJECT_ROOT / ".env.example"
        if example.exists():
            shutil.copyfile(example, env_path)
        else:
            env_path.write_text("")
    text = env_path.read_text()

    existing = _KEY_LINE.search(text)
    if existing and existing.group(1).strip():
        sys.exit(
            f"{env_path} already has a SOLANA_PRIVATE_KEY. Not replacing it: the bot "
            "would lose access to that wallet and anything in it. Move the funds "
            "out and remove the line first if you really want a new one."
        )

    keypair = Keypair()
    address = str(keypair.pubkey())

    backup = BACKUP_DIR / f"wallet-{address}.json"
    _write_private(backup, json.dumps(list(bytes(keypair))))

    line = f"SOLANA_PRIVATE_KEY={keypair}"
    if existing:
        text = _KEY_LINE.sub(lambda _m: line, text, count=1)
    else:
        text = text + ("" if text.endswith("\n") or not text else "\n") + line + "\n"
    _write_private(env_path, text)

    print(f"New bot wallet: {address}")
    print(f"  key written to {env_path} (SOLANA_PRIVATE_KEY)")
    print(f"  backup saved to {backup}")
    print(
        "\nSend only what you can afford to lose to that address. Keep the backup\n"
        "somewhere safe: anyone with it can spend the wallet, and without it and\n"
        ".env the funds are gone."
    )


if __name__ == "__main__":
    main()
