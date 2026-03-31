import argparse
import os

from stock_recommender.data.database import DatabaseManager
from stock_recommender.data.universe_downloader import (
    IndianUniverseDownloader,
    load_nse_universe_official,
    load_universe_from_csv,
    write_universe_metadata,
)


def main():
    parser = argparse.ArgumentParser(
        description="Download maximum available historical OHLCV for Indian stock universe symbols"
    )
    parser.add_argument("--symbols-file", default=None, help="CSV master list with stock symbols")
    parser.add_argument("--auto-universe", action="store_true", help="Fetch official NSE universe automatically")
    parser.add_argument("--exchange", default="NSE", choices=["NSE", "BSE"], help="Exchange for symbols file")
    parser.add_argument("--output-dir", default="india_universe_data", help="Where per-symbol CSV files will be saved")
    parser.add_argument("--symbol-col", default=None, help="Optional symbol column override")
    parser.add_argument("--name-col", default=None, help="Optional company-name column override")
    parser.add_argument("--sector-col", default=None, help="Optional sector column override")
    parser.add_argument("--workers", type=int, default=4, help="Parallel downloads")
    parser.add_argument("--pause-seconds", type=float, default=0.5, help="Pause between provider calls")
    parser.add_argument("--period", default="max", help="History period, default max")
    parser.add_argument("--min-rows", type=int, default=30, help="Minimum rows required to keep a symbol")
    parser.add_argument("--retries", type=int, default=2, help="Retries per symbol")
    parser.add_argument("--refresh", action="store_true", help="Redownload even if CSV already exists")
    parser.add_argument("--metadata-out", default=None, help="Optional JSON metadata file for later ingestion")
    parser.add_argument("--db", default=None, help="Optional SQLite DB path to ingest after download")
    args = parser.parse_args()

    if args.auto_universe:
        if args.exchange != "NSE":
            raise SystemExit("Automatic universe fetch is currently implemented only for official NSE universe")
        symbols = load_nse_universe_official()
    else:
        if not args.symbols_file:
            raise SystemExit("--symbols-file is required unless --auto-universe is used")
        symbols = load_universe_from_csv(
            path=args.symbols_file,
            exchange=args.exchange,
            symbol_col=args.symbol_col,
            name_col=args.name_col,
            sector_col=args.sector_col,
        )
    if not symbols:
        raise SystemExit("No symbols found in the supplied master file")

    downloader = IndianUniverseDownloader(
        output_dir=args.output_dir,
        workers=args.workers,
        pause_seconds=args.pause_seconds,
        verbose=True,
    )
    report = downloader.download(
        symbols=symbols,
        period=args.period,
        refresh=args.refresh,
        min_rows=args.min_rows,
        retries=args.retries,
    )

    metadata_out = args.metadata_out or os.path.join(args.output_dir, "_metadata.json")
    write_universe_metadata(metadata_out, symbols)

    print("\nDOWNLOAD REPORT")
    print("-" * 60)
    print(f"Requested             : {report.requested}")
    print(f"Downloaded            : {report.downloaded}")
    print(f"Skipped existing      : {report.skipped}")
    print(f"Failed                : {report.failed}")
    print(f"Metadata file         : {metadata_out}")
    if report.failures:
        print("First failures        :")
        for item in report.failures[:20]:
            print(f"  {item}")

    if args.db:
        db = DatabaseManager(args.db)
        ingest_report = downloader.ingest_to_db(
            db=db,
            metadata_path=metadata_out,
            market_prefix=args.exchange,
        )
        print("\nDB INGEST REPORT")
        print("-" * 60)
        print(f"Imported stocks       : {ingest_report.imported_stocks}")
        print(f"Imported rows         : {ingest_report.imported_rows}")
        print(f"Skipped files         : {len(ingest_report.skipped_files)}")


if __name__ == "__main__":
    main()
