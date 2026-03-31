"""
Bulk-download NSE F&O archive files with resume support and a progress bar.

This script drives the official NSE archive UI because older derivatives
archives are not exposed through a single stable public URL pattern.

Default behavior:
  - iterate weekday dates in the requested range
  - for each date, select the F&O archive report type
  - download the matching file(s) into the target directory
  - record status in a JSONL manifest so interrupted runs can resume

Examples
--------
python scripts/download_nse_fo_archives.py ^
  --start 2000-01-01 ^
  --end 2026-03-28 ^
  --output-dir data\\nse_fo_archives

python scripts/download_nse_fo_archives.py ^
  --start 2018-01-01 ^
  --end 2024-07-05 ^
  --report-label "Bhavcopy (fo.zip)" ^
  --browser edge

Notes
-----
1. Default preset uses:
     - "Bhavcopy (fo.zip)" before 2024-07-08
     - "UDiFF Common Bhavcopy Final (zip)" on/after 2024-07-08
2. The NSE page may occasionally change its DOM. If that happens, adjust the
   selectors in `_open_archive_panel()` and `_select_archive_report()`.
3. NSE may not have files for weekends, exchange holidays, or some early dates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

from tqdm import tqdm
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver import ChromeOptions, EdgeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver, WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait


ARCHIVE_URL = "https://www.nseindia.com/all-reports-derivatives"
UDIFF_CUTOVER = date(2024, 7, 8)
DEFAULT_TIMEOUT = 30


@dataclass(frozen=True)
class DownloadPlan:
    trade_date: date
    report_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NSE F&O archive files in bulk.")
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument(
        "--output-dir",
        default="data/nse_fo_archives",
        help="Directory where files and manifest will be stored.",
    )
    parser.add_argument(
        "--browser",
        choices=["edge", "chrome"],
        default="edge",
        help="Browser for Selenium automation.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode. Off by default because NSE is stricter with headless sessions.",
    )
    parser.add_argument(
        "--report-mode",
        choices=["bhavcopy_auto", "fixed_label"],
        default="bhavcopy_auto",
        help="Use automatic pre/post-UDiFF bhavcopy switching or a fixed report label.",
    )
    parser.add_argument(
        "--report-label",
        default=None,
        help='Archive dropdown label to use when --report-mode=fixed_label, for example "Bhavcopy (fo.zip)".',
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Per-action Selenium timeout in seconds.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="Small pause between dates so the site is not hammered.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip dates already marked as downloaded or missing in the manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = parse_iso_date(args.start)
    end = parse_iso_date(args.end)
    if end < start:
        raise SystemExit("--end must be on or after --start")
    if args.report_mode == "fixed_label" and not args.report_label:
        raise SystemExit("--report-label is required when --report-mode=fixed_label")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    manifest = load_manifest(manifest_path)

    plans = list(build_download_plans(start, end, args.report_mode, args.report_label))
    if args.resume:
        plans = [plan for plan in plans if not is_completed(manifest, plan)]

    if not plans:
        print("Nothing to do.")
        return 0

    driver = build_driver(args.browser, output_dir, args.headless)
    wait = WebDriverWait(driver, args.timeout)

    try:
        open_archive_panel(driver, wait)
        progress = tqdm(plans, desc="NSE F&O archives", unit="day")
        for plan in progress:
            progress.set_postfix_str(f"{plan.trade_date.isoformat()} | {plan.report_label}")
            try:
                downloaded = process_date(driver, wait, output_dir, plan)
                if downloaded:
                    append_manifest(manifest_path, plan, "downloaded", downloaded.name)
                else:
                    append_manifest(manifest_path, plan, "missing", "")
            except Exception as exc:  # pragma: no cover - operational path
                append_manifest(manifest_path, plan, "error", str(exc))
                print(f"[ERROR] {plan.trade_date} {plan.report_label}: {exc}", file=sys.stderr)
            time.sleep(args.settle_seconds)
    finally:
        driver.quit()

    return 0


def parse_iso_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def build_download_plans(
    start: date,
    end: date,
    report_mode: str,
    fixed_label: Optional[str],
) -> Iterable[DownloadPlan]:
    current = start
    while current <= end:
        if current.weekday() < 5:
            label = resolve_report_label(current, report_mode, fixed_label)
            yield DownloadPlan(trade_date=current, report_label=label)
        current += timedelta(days=1)


def resolve_report_label(trade_date: date, report_mode: str, fixed_label: Optional[str]) -> str:
    if report_mode == "fixed_label":
        assert fixed_label is not None
        return fixed_label
    if trade_date >= UDIFF_CUTOVER:
        return "UDiFF Common Bhavcopy Final (zip)"
    return "Bhavcopy (fo.zip)"


def load_manifest(path: Path) -> dict[tuple[str, str], dict]:
    entries: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            entries[(item["trade_date"], item["report_label"])] = item
    return entries


def is_completed(manifest: dict[tuple[str, str], dict], plan: DownloadPlan) -> bool:
    item = manifest.get((plan.trade_date.isoformat(), plan.report_label))
    return bool(item and item.get("status") in {"downloaded", "missing"})


def append_manifest(path: Path, plan: DownloadPlan, status: str, detail: str) -> None:
    record = {
        "trade_date": plan.trade_date.isoformat(),
        "report_label": plan.report_label,
        "status": status,
        "detail": detail,
        "recorded_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_driver(browser: str, download_dir: Path, headless: bool) -> WebDriver:
    selenium_cache = (download_dir / ".selenium-cache").resolve()
    selenium_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SE_CACHE_PATH", str(selenium_cache))
    os.environ.setdefault("SE_AVOID_STATS", "true")

    prefs = {
        "download.default_directory": str(download_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }

    if browser == "chrome":
        options = ChromeOptions()
        options.add_experimental_option("prefs", prefs)
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--window-size=1600,1200")
        if headless:
            options.add_argument("--headless=new")
        return webdriver.Chrome(options=options)

    options = EdgeOptions()
    options.use_chromium = True
    options.add_experimental_option("prefs", prefs)
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1600,1200")
    if headless:
        options.add_argument("--headless=new")
    return webdriver.Edge(options=options)


def open_archive_panel(driver: WebDriver, wait: WebDriverWait) -> None:
    driver.get(ARCHIVE_URL)
    dismiss_cookie_banner(driver)

    # Keep the navigation steps explicit. The page uses tabs and can render
    # multiple archive sections; we want the F&O one.
    click_by_text(driver, wait, "Historical Reports")
    click_by_text(driver, wait, "Equity Derivatives")
    try_click_by_text(driver, wait, "Archives", timeout_seconds=8)
    wait.until(lambda d: has_visible_text(d, "Select Report"))


def dismiss_cookie_banner(driver: WebDriver) -> None:
    for label in ("Accept", "I Accept", "Agree", "OK"):
        try:
            element = first_visible(driver, f"//*[self::button or self::a][contains(normalize-space(.), {xpath_literal(label)})]")
            if element is not None:
                safe_click(driver, element)
                time.sleep(0.5)
                return
        except Exception:
            continue


def process_date(
    driver: WebDriver,
    wait: WebDriverWait,
    output_dir: Path,
    plan: DownloadPlan,
) -> Optional[Path]:
    panel = locate_archive_panel(driver, wait, plan.report_label)
    select_archive_report(driver, panel, plan.report_label)
    set_archive_date(driver, panel, plan.trade_date)

    before = snapshot_download_dir(output_dir)
    click_download_target(driver, wait, panel, plan.report_label)
    return wait_for_new_download(output_dir, before, timeout=wait._timeout)


def locate_archive_panel(driver: WebDriver, wait: WebDriverWait, report_label: str) -> WebElement:
    wait.until(lambda d: has_visible_text(d, "Select Report"))
    panels = driver.find_elements(
        By.XPATH,
        "//*[.//*[contains(normalize-space(.), 'Select Report')] and .//*[contains(normalize-space(.), 'Date') or contains(normalize-space(.), 'From')]]",
    )
    visible_panels = [panel for panel in panels if panel.is_displayed()]
    if len(visible_panels) == 1:
        return visible_panels[0]

    # Prefer the panel whose report dropdown or body contains the requested label.
    for panel in visible_panels:
        if report_label.lower() in panel.text.lower():
            return panel

    # Fallback: first visible archive panel after tab activation.
    if visible_panels:
        return visible_panels[0]
    raise TimeoutException("Could not locate the F&O archive panel")


def select_archive_report(driver: WebDriver, panel: WebElement, report_label: str) -> None:
    selects = [el for el in panel.find_elements(By.TAG_NAME, "select") if el.is_displayed()]
    if selects:
        try:
            Select(selects[0]).select_by_visible_text(report_label)
            return
        except Exception:
            pass

    # Custom dropdown fallback.
    dropdown_candidates = panel.find_elements(
        By.XPATH,
        ".//*[contains(@class, 'dropdown') or contains(@class, 'select') or @role='combobox']",
    )
    for dropdown in dropdown_candidates:
        if not dropdown.is_displayed():
            continue
        safe_click(driver, dropdown)
        option = first_visible(driver, f"//*[contains(normalize-space(.), {xpath_literal(report_label)})]")
        if option is not None:
            safe_click(driver, option)
            return

    raise NoSuchElementException(f"Could not select archive report: {report_label}")


def set_archive_date(driver: WebDriver, panel: WebElement, trade_date: date) -> None:
    value_candidates = [
        trade_date.strftime("%d-%b-%Y"),
        trade_date.strftime("%d-%m-%Y"),
        trade_date.strftime("%d/%m/%Y"),
    ]
    inputs = [
        el for el in panel.find_elements(By.XPATH, ".//input[not(@type='hidden')]")
        if el.is_displayed() and el.is_enabled()
    ]
    if not inputs:
        raise NoSuchElementException("Could not find archive date input")

    date_input = inputs[0]
    for value in value_candidates:
        try:
            driver.execute_script(
                """
                arguments[0].removeAttribute('readonly');
                arguments[0].value = '';
                arguments[0].focus();
                arguments[0].value = arguments[1];
                arguments[0].dispatchEvent(new Event('input', {bubbles: true}));
                arguments[0].dispatchEvent(new Event('change', {bubbles: true}));
                arguments[0].blur();
                """,
                date_input,
                value,
            )
            time.sleep(1.0)
            return
        except Exception:
            continue
    raise RuntimeError(f"Could not set archive date for {trade_date.isoformat()}")


def click_download_target(
    driver: WebDriver,
    wait: WebDriverWait,
    panel: WebElement,
    report_label: str,
) -> None:
    # Give the page a chance to render the downloadable artifact for the selected date.
    time.sleep(2.0)

    link = first_visible(
        panel,
        ".//a[contains(@href, '.zip') or contains(@href, '.csv') or contains(@href, '.gz') or contains(@href, '.dat') or contains(normalize-space(.), '.zip') or contains(normalize-space(.), '.csv') or contains(normalize-space(.), '.gz') or contains(normalize-space(.), '.DAT')]",
    )
    if link is not None:
        safe_click(driver, link)
        return

    # Fallback to the multiple-file download control, which works when the page
    # renders one or more selected files under the archive panel.
    button = first_visible(
        panel,
        ".//*[self::button or self::a or @role='button'][contains(normalize-space(.), 'Multiple file Download')]",
    )
    if button is not None:
        safe_click(driver, button)
        return

    raise RuntimeError(f"No downloadable file appeared for report '{report_label}'")


def snapshot_download_dir(path: Path) -> set[tuple[str, int, int]]:
    snapshot: set[tuple[str, int, int]] = set()
    for item in path.iterdir():
        if item.is_file():
            stat = item.stat()
            snapshot.add((item.name, stat.st_size, int(stat.st_mtime)))
    return snapshot


def wait_for_new_download(path: Path, before: set[tuple[str, int, int]], timeout: float) -> Optional[Path]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        partials = []
        candidates = []
        for item in path.iterdir():
            if not item.is_file():
                continue
            suffix = item.suffix.lower()
            if suffix in {".crdownload", ".part", ".tmp"}:
                partials.append(item)
                continue
            stat = item.stat()
            marker = (item.name, stat.st_size, int(stat.st_mtime))
            if marker not in before:
                candidates.append(item)
        if candidates and not partials:
            return max(candidates, key=lambda p: p.stat().st_mtime)
        time.sleep(1.0)
    return None


def click_by_text(driver: WebDriver, wait: WebDriverWait, text: str) -> None:
    literal = xpath_literal(text)
    xpath = (
        f"//*[self::a or self::button or @role='tab' or @role='button']"
        f"[contains(normalize-space(.), {literal})]"
    )
    wait.until(lambda d: first_visible(d, xpath) is not None)
    element = first_visible(driver, xpath)
    if element is None:
        raise TimeoutException(f"Could not find clickable text: {text}")
    safe_click(driver, element)
    time.sleep(1.0)


def try_click_by_text(driver: WebDriver, wait: WebDriverWait, text: str, timeout_seconds: int = 5) -> bool:
    literal = xpath_literal(text)
    xpath = (
        f"//*[self::a or self::button or @role='tab' or @role='button']"
        f"[contains(normalize-space(.), {literal})]"
    )
    short_wait = WebDriverWait(driver, timeout_seconds)
    try:
        short_wait.until(lambda d: first_visible(d, xpath) is not None)
    except TimeoutException:
        return False
    element = first_visible(driver, xpath)
    if element is None:
        return False
    safe_click(driver, element)
    time.sleep(1.0)
    return True


def safe_click(driver: WebDriver, element: WebElement) -> None:
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
    try:
        element.click()
    except Exception:
        driver.execute_script("arguments[0].click();", element)


def has_visible_text(driver: WebDriver, text: str) -> bool:
    return first_visible(driver, f"//*[contains(normalize-space(.), {xpath_literal(text)})]") is not None


def first_visible(root: WebDriver | WebElement, xpath: str) -> Optional[WebElement]:
    elements = root.find_elements(By.XPATH, xpath)
    for element in elements:
        try:
            if element.is_displayed():
                return element
        except Exception:
            continue
    return None


def xpath_literal(text: str) -> str:
    if "'" not in text:
        return f"'{text}'"
    if '"' not in text:
        return f'"{text}"'
    parts = text.split("'")
    return "concat(" + ", \"'\", ".join(f"'{part}'" for part in parts) + ")"


if __name__ == "__main__":
    raise SystemExit(main())
