"""FNSPID historical OHLCV.

FNSPID ships prices as `Stock_price/full_history.zip` (~0.59 GB): one CSV per
ticker, 7,693 tickers, with the schema

    date,open,high,low,close,adj close,volume

rows in DESCENDING date order, reaching back to the 1980s for older names.

Adjustment
----------
The archive carries BOTH a raw `close` and an `adj close`. The existing
yfinance source uses ``auto_adjust=True``, whose `close` is already fully
adjusted for splits and dividends. Taking FNSPID's raw `close` instead would
manufacture a fake return on every split and dividend date.

So this source reconstructs a fully-adjusted bar: it derives the per-row
factor ``adj close / close`` and applies it to open/high/low/close. Volume is
left untouched, matching the yfinance convention (Yahoo already reports
split-adjusted volume, and ``auto_adjust`` does not rescale it).

This mirrors yfinance's adjustment convention. Note the standard caveat: an
adjusted price at date T embeds corporate actions that happened AFTER T.

The adjustment seam
-------------------
The archive splices two differently-adjusted vintages at a single shared
date. Rows BEFORE the seam are not back-adjusted for corporate actions that
occurred after it; rows from the seam onward are. For any ticker that split
after the seam this produces a synthetic one-day crash, e.g.

    AAPL  close 364.11 -> 91.52 on 2020-07-06   (-74.9%, the 4:1 split)
    AMZN  -94.7%  (20:1)      TSLA  -92.4%  (5:1 then 3:1)

The seam date was established empirically, not assumed: scanning all 7,549
readable tickers, 2020-07-06 is the single most common date for a >30% move
(491 tickers) despite being an unremarkable session -- ahead of every day of
the COVID crash. See ``SEAM_DATE``.

``repair_seam`` rescales the pre-seam segment back onto the post-seam basis so
returns and rolling indicators are continuous. It is applied by default; pass
``repair=False`` to read the archive verbatim.

Deliberately NOT attempted here:

* A general "detect every unadjusted corporate action" repair. Scanning for
  large overnight gaps sitting on canonical split ratios flags 25,622 events
  across 2,416 tickers, the vast majority of which are ordinary volatility in
  low-priced names. Rewriting that much genuine price history to chase a few
  real splits is far worse than the defect.
* Repairing tickers whose file ENDS before a later split. GOOGL's archive data
  stops at 2020-04-01, so its whole series predates the 2022 20:1 split and
  sits on a stale basis with no seam inside the file to key on. Nothing in the
  archive reveals the missing factor, so these are left untouched and reported
  rather than guessed at.
"""

from __future__ import annotations

import io
import os
import ssl
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from app.data.sources.base import REQUIRED_OHLCV

FNSPID_REPO = "Zihan1004/FNSPID"
FNSPID_REVISION = "main"
FNSPID_PRICE_FILE = "Stock_price/full_history.zip"
FNSPID_PRICE_URL = (
    f"https://huggingface.co/datasets/{FNSPID_REPO}/resolve/"
    f"{FNSPID_REVISION}/{FNSPID_PRICE_FILE}"
)

DEFAULT_PRICE_ZIP = Path("data/raw/prices/fnspid_full_history.zip")

MEMBER_PREFIX = "full_history/"
RAW_COLUMNS = ["date", "open", "high", "low", "close", "adj close", "volume"]
ADJ_CLOSE = "adj close"
# macOS resource forks are archived alongside the real CSVs; they are not data.
_JUNK_PREFIX = "__MACOSX"

# Where the archive stitches its two adjustment vintages. Empirical: across
# 7,549 tickers this is the most common date for a >30% single-day move (491
# tickers), on a session that was otherwise unremarkable.
SEAM_DATE = pd.Timestamp("2020-07-06")

# A seam has to be a big, persistent level shift before we touch anything.
MIN_SEAM_GAP = 1.30       # >=30% overnight gap in either direction
PERSISTENCE_WINDOW = 10   # sessions used to confirm the shift does not revert
PERSISTENCE_TOL = 0.25    # post-seam level must hold within 25% of the shift


def member_name(ticker: str) -> str:
    return f"{MEMBER_PREFIX}{ticker.strip().upper()}.csv"


def apply_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """Back-adjust OHLC by ``adj close / close``.

    Pure and network-free so the adjustment rule can be tested directly.
    Rows whose factor cannot be computed (missing or non-positive `close`,
    missing `adj close`) fall back to a factor of 1.0 rather than producing
    inf/NaN prices; validation downstream still screens the result.
    """
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    adj = pd.to_numeric(out[ADJ_CLOSE], errors="coerce")

    with np.errstate(invalid="ignore", divide="ignore"):
        factor = adj / close.where(close > 0)
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce") * factor
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    return out


def detect_seam(df: pd.DataFrame, seam_date: pd.Timestamp = SEAM_DATE) -> float | None:
    """Return the multiplicative factor by which the pre-seam segment is off.

    Returns None when the ticker is unaffected -- no seam inside the file, too
    little data around it, the gap is small, or the shift reverts (which means
    it was a spike, not a rescale).

    The factor is the observed overnight gap ``close[D-1] / open[D]``, NOT a
    canonical split ratio snapped to 2/4/20. The seam combines the split with
    the difference in accumulated dividend adjustment between the two
    vintages, so the true factor is deliberately not a round number: AAPL's is
    ~4.02, not 4.00. Snapping would leave a systematic residual.
    """
    if len(df) < 2:
        return None
    idx = df.index
    pos = idx.searchsorted(seam_date)
    # Seam must fall strictly inside the file, with a row on each side.
    if pos <= 0 or pos >= len(df):
        return None

    prev_close = float(df["close"].iloc[pos - 1])
    day_open = float(df["open"].iloc[pos])
    if not (np.isfinite(prev_close) and np.isfinite(day_open)) or day_open <= 0:
        return None

    gap = prev_close / day_open
    if not np.isfinite(gap) or gap <= 0:
        return None
    if 1 / MIN_SEAM_GAP < gap < MIN_SEAM_GAP:
        return None

    # Persistence: a true rescale holds; a one-day glitch snaps back.
    after = df["close"].iloc[pos : pos + PERSISTENCE_WINDOW]
    if len(after) < 2:
        return None
    level_after = float(np.nanmedian(after.to_numpy(dtype=float)))
    if not np.isfinite(level_after) or level_after <= 0:
        return None
    implied = prev_close / level_after
    if abs(np.log(implied / gap)) > np.log(1 + PERSISTENCE_TOL):
        return None

    return gap


def repair_seam(
    df: pd.DataFrame, seam_date: pd.Timestamp = SEAM_DATE
) -> tuple[pd.DataFrame, float | None]:
    """Rescale the pre-seam segment onto the post-seam basis.

    Divides open/high/low/close before `seam_date` by the detected factor.
    Volume is NOT rescaled: share counts are a separate series and the archive
    already reports post-split volume on both sides of the seam, so touching it
    would introduce the very discontinuity we are removing.

    Deterministic and idempotent-in-spirit: a repaired frame no longer has a
    detectable seam, so re-running is a no-op.
    """
    factor = detect_seam(df, seam_date)
    if factor is None:
        return df, None
    out = df.copy()
    mask = out.index < seam_date
    out.loc[mask, ["open", "high", "low", "close"]] /= factor
    return out, factor


def parse_price_csv(raw: bytes | str, repair: bool = True) -> pd.DataFrame:
    """Parse one FNSPID ticker CSV into an adjusted, ascending OHLCV frame.

    With ``repair=True`` (default) the archive's adjustment seam is corrected.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(raw))
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"FNSPID price CSV missing columns {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]
    df = apply_adjustment(df)

    df = df.set_index("date").sort_index()  # archive stores newest-first
    df.index = df.index.tz_localize(None).normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")]
    df = df[REQUIRED_OHLCV]
    if repair:
        df, _ = repair_seam(df)
    return df


def download_fnspid_prices(
    dest: str | Path = DEFAULT_PRICE_ZIP,
    url: str = FNSPID_PRICE_URL,
    chunk_bytes: int = 1 << 22,
    progress: bool = True,
) -> Path:
    """Fetch the price archive into the gitignored data/ cache, resumably."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if progress:
            print(f"[INFO] FNSPID prices already cached ({dest.stat().st_size / 1e9:.2f} GB)")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": "ai-market-intelligence/0.1"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)

    ctx = ssl.create_default_context()
    started, seen = time.time(), have
    with urllib.request.urlopen(req, timeout=180, context=ctx) as resp, open(tmp, "ab") as fh:
        total = int(resp.headers.get("Content-Length", 0)) + have
        while True:
            block = resp.read(chunk_bytes)
            if not block:
                break
            fh.write(block)
            seen += len(block)
            if progress and seen % (1 << 27) < chunk_bytes:
                rate = seen / 1e6 / max(time.time() - started, 1e-9)
                print(f"     {seen / 1e9:.2f}/{total / 1e9:.2f} GB  {rate:.1f} MB/s", flush=True)
    os.replace(tmp, dest)
    return dest


class _RemoteZipFile(io.RawIOBase):
    """Seekable file-like over an HTTP resource using Range requests.

    Lets us read a single ticker out of the archive without downloading all
    0.59 GB. The zip central directory lives at the end of the file, so the
    tail is fetched once and cached; everything else is ranged on demand.
    """

    def __init__(self, url: str, tail_bytes: int = 6 << 20, timeout: int = 120):
        self.url = url
        self.timeout = timeout
        self._ctx = ssl.create_default_context()
        self._pos = 0
        head = urllib.request.Request(
            url, headers={"User-Agent": "ai-market-intelligence/0.1"}, method="HEAD"
        )
        with urllib.request.urlopen(head, timeout=timeout, context=self._ctx) as resp:
            self.size = int(resp.headers["Content-Length"])
        self._tail_start = max(0, self.size - tail_bytes)
        self._tail = self._range(self._tail_start, self.size - 1)

    def _range(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(
            self.url,
            headers={
                "User-Agent": "ai-market-intelligence/0.1",
                "Range": f"bytes={start}-{end}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx) as resp:
            return resp.read()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        else:
            self._pos = self.size + offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readinto(self, buf) -> int:
        start, end = self._pos, min(self._pos + len(buf), self.size)
        if start >= end:
            return 0
        if start >= self._tail_start:
            data = self._tail[start - self._tail_start : end - self._tail_start]
        else:
            data = self._range(start, end - 1)
        buf[: len(data)] = data
        self._pos = start + len(data)
        return len(data)


class FnspidPriceSource:
    """PriceSource over the FNSPID price archive.

    Satisfies the same ``fetch(ticker, start, end)`` contract as
    YFinanceSource and returns an identically shaped frame: tz-naive
    normalized DatetimeIndex named 'date', columns open/high/low/close/volume,
    ascending, fully adjusted.

    Reads a locally cached zip when available; otherwise streams individual
    members over HTTP range requests.
    """

    name = "fnspid"

    def __init__(
        self,
        zip_path: str | Path = DEFAULT_PRICE_ZIP,
        url: str = FNSPID_PRICE_URL,
        repair: bool = True,
    ):
        self.zip_path = Path(zip_path)
        self.url = url
        self.repair = repair
        self._zf: zipfile.ZipFile | None = None

    def available(self) -> bool:
        return self.zip_path.exists()

    def _archive(self) -> zipfile.ZipFile:
        if self._zf is None:
            if self.zip_path.exists():
                self._zf = zipfile.ZipFile(self.zip_path)
            else:
                self._zf = zipfile.ZipFile(
                    io.BufferedReader(_RemoteZipFile(self.url), buffer_size=1 << 20)
                )
        return self._zf

    def tickers(self) -> list[str]:
        """Every ticker in the archive, excluding macOS resource forks."""
        names = self._archive().namelist()
        return sorted(
            n[len(MEMBER_PREFIX) : -4]
            for n in names
            if n.startswith(MEMBER_PREFIX) and n.endswith(".csv")
            and not n.startswith(_JUNK_PREFIX)
        )

    def has_ticker(self, ticker: str) -> bool:
        try:
            self._archive().getinfo(member_name(ticker))
        except KeyError:
            return False
        return True

    def fetch(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Adjusted OHLCV for one ticker within [start, end].

        Returns an empty, correctly-columned frame when the ticker is absent
        from the archive -- NVDA, for instance, is not present -- so callers
        behave the same as they do for an empty yfinance response.

        The adjustment seam is repaired unless the source was constructed with
        ``repair=False``.
        """
        try:
            raw = self._archive().read(member_name(ticker))
        except KeyError:
            return pd.DataFrame(
                columns=REQUIRED_OHLCV,
                index=pd.DatetimeIndex([], name="date"),
            )

        # Repair BEFORE date filtering: a window starting after the seam
        # would otherwise hide the discontinuity needed to detect it.
        df = parse_price_csv(raw, repair=self.repair)
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        return df

    def close(self) -> None:
        if self._zf is not None:
            self._zf.close()
            self._zf = None
