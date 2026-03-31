import json
from pathlib import Path


STATE_PATH = Path(r"D:\AI\hackathon\india_universe_data\_download_state.json")
OUT_PATH = Path(r"D:\AI\hackathon\failed_nse_symbols.csv")


def main():
    data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    failed = sorted(
        key.split(":")[-1]
        for key, value in data.items()
        if isinstance(value, dict) and value.get("status") == "failed"
    )
    OUT_PATH.write_text(
        "Symbol\n" + "\n".join(failed) + ("\n" if failed else ""),
        encoding="utf-8",
    )
    print(f"Wrote {len(failed)} symbols to {OUT_PATH}")


if __name__ == "__main__":
    main()
