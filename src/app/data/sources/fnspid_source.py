"""FNSPID historical financial news.

FNSPID (Financial News and Stock Price Integration Dataset) is a frozen public
dataset on HuggingFace covering 1999-2023. It is used here because it carries
an explicit `Stock_symbol` per article: ticker attribution comes from the
dataset itself, never from fuzzy title/organization matching.

Being frozen is a feature -- pinning the revision makes ingestion reproducible
byte-for-byte, which no live news API can offer.

License: CC BY-NC-4.0 (non-commercial). Educational research use only.

Pipeline:

    download_fnspid()        5.73 GB raw CSV -> data/raw/news/   (gitignored)
    extract_universe()       chunked filter  -> small parquet
    FnspidNewsSource.fetch() parquet         -> canonical news schema
"""

from __future__ import annotations

import hashlib
import io
import os
import ssl
import time
import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from app.data.sources.base import NEWS_SCHEMA

FNSPID_REPO = "Zihan1004/FNSPID"
FNSPID_REVISION = "main"


def _url(filename: str) -> str:
    return (
        f"https://huggingface.co/datasets/{FNSPID_REPO}/resolve/{FNSPID_REVISION}/{filename}"
    )


# All_external.csv looks convenient (5.73 GB, no article bodies) but only ~25%
# of its rows carry a Stock_symbol, MSFT is absent entirely, and coverage stops
# in 2020-06. nasdaq_exteral_data.csv is 100% attributed and reaches 2023-12,
# which is why it is the source of record despite being 23 GB.
FNSPID_FILE = "Stock_news/All_external.csv"
FNSPID_URL = _url(FNSPID_FILE)

FNSPID_NASDAQ_FILE = "Stock_news/nasdaq_exteral_data.csv"  # sic: upstream spelling
FNSPID_NASDAQ_URL = _url(FNSPID_NASDAQ_FILE)

# Verified against the real file, not the documentation.
FNSPID_COLUMNS = [
    "Date",
    "Article_title",
    "Stock_symbol",
    "Url",
    "Publisher",
    "Author",
    "Article",
    "Lsa_summary",
    "Luhn_summary",
    "Textrank_summary",
    "Lexrank_summary",
]
USED_COLUMNS = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher"]

# Every row is "YYYY-MM-DD HH:MM:SS UTC". Parsed strictly so a format drift
# surfaces as NaT (and is counted) instead of being silently coerced.
FNSPID_DATE_FORMAT = "%Y-%m-%d %H:%M:%S UTC"

DEFAULT_RAW_PATH = Path("data/raw/news/fnspid_all_external.csv")
DEFAULT_PARQUET_PATH = Path("data/curated/news_fnspid.parquet")


def download_fnspid(
    dest: str | Path = DEFAULT_RAW_PATH,
    url: str = FNSPID_URL,
    chunk_bytes: int = 1 << 22,
    progress: bool = True,
) -> Path:
    """Download the raw CSV, resuming a partial `.part` file if present.

    The raw file lives under data/ which is gitignored -- it is a cache, not
    a project artifact, and is never committed.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if progress:
            print(f"[INFO] raw FNSPID already present ({dest.stat().st_size / 1e9:.2f} GB)")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "ai-market-intelligence/0.1"})
    if have:
        req.add_header("Range", f"bytes={have}-")

    ctx = ssl.create_default_context()
    start, seen = time.time(), have
    with urllib.request.urlopen(req, timeout=120, context=ctx) as resp, open(tmp, "ab") as fh:
        total = int(resp.headers.get("Content-Length", 0)) + have
        while True:
            block = resp.read(chunk_bytes)
            if not block:
                break
            fh.write(block)
            seen += len(block)
            if progress and seen % (1 << 28) < chunk_bytes:
                rate = seen / 1e6 / max(time.time() - start, 1e-9)
                print(f"     {seen / 1e9:.2f}/{total / 1e9:.2f} GB  {rate:.1f} MB/s", flush=True)
    os.replace(tmp, dest)
    return dest


def extract_universe(
    tickers: Sequence[str],
    raw_path: str | Path = DEFAULT_RAW_PATH,
    out_path: str | Path = DEFAULT_PARQUET_PATH,
    chunksize: int = 500_000,
    progress: bool = True,
) -> dict:
    """Chunk-filter the 5.73 GB CSV down to our universe.

    Streams through the file in chunks so peak RAM stays proportional to
    `chunksize`, never the file size. Returns ingestion statistics.
    """
    raw_path, out_path = Path(raw_path), Path(out_path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found; run download_fnspid() (or `cli ingest-news --download`) first"
        )
    wanted = {t.strip().upper() for t in tickers}

    kept: list[pd.DataFrame] = []
    rows_scanned = 0
    bad_symbol = 0
    reader = pd.read_csv(
        raw_path,
        usecols=USED_COLUMNS,
        chunksize=chunksize,
        dtype=str,
        low_memory=False,
    )
    for i, chunk in enumerate(reader):
        rows_scanned += len(chunk)
        sym = chunk["Stock_symbol"].astype("string").str.strip().str.upper()
        bad_symbol += int(sym.isna().sum())
        hit = chunk[sym.isin(wanted)].copy()
        if len(hit):
            hit["Stock_symbol"] = sym[sym.isin(wanted)]
            kept.append(hit)
        if progress and i % 10 == 0:
            n = sum(len(k) for k in kept)
            print(f"     scanned {rows_scanned:>12,}  kept {n:>9,}", flush=True)

    if not kept:
        raise ValueError(f"no FNSPID rows matched {sorted(wanted)}")

    df = pd.concat(kept, ignore_index=True)
    matched = len(df)

    # Strict parse: a format drift becomes NaT and is reported, not coerced.
    published = pd.to_datetime(
        df["Date"], format=FNSPID_DATE_FORMAT, utc=True, errors="coerce"
    )
    n_invalid = int(published.isna().sum())
    df["published_at"] = published
    df = df[df["published_at"].notna()].copy()

    out = pd.DataFrame(
        {
            "headline": df["Article_title"].astype("string").str.strip(),
            "source": df["Publisher"].astype("string").fillna("unknown"),
            "url": df["Url"].astype("string").fillna(""),
            "published_at": df["published_at"],
            "ticker": df["Stock_symbol"].astype("string"),
        }
    )
    out = out[out["headline"].notna() & (out["headline"].str.len() > 0)]
    out["article_id"] = [
        hashlib.sha1(f"{u}|{h}|{t}".encode()).hexdigest()[:16]
        for u, h, t in zip(out["url"], out["headline"], out["ticker"], strict=True)
    ]
    before_dedupe = len(out)
    out = out.drop_duplicates(subset=["article_id"]).reset_index(drop=True)

    out = out[NEWS_SCHEMA].sort_values(["ticker", "published_at"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    return {
        "rows_scanned": rows_scanned,
        "rows_matched": matched,
        "rows_invalid_timestamp": n_invalid,
        "rows_blank_symbol": bad_symbol,
        "duplicates_removed": before_dedupe - len(out),
        "rows_written": len(out),
        "out_path": str(out_path),
        "bytes_written": out_path.stat().st_size,
        "earliest": out["published_at"].min(),
        "latest": out["published_at"].max(),
    }


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw FNSPID columns onto the canonical news schema."""
    published = pd.to_datetime(
        df["Date"], format=FNSPID_DATE_FORMAT, utc=True, errors="coerce"
    )
    out = pd.DataFrame(
        {
            "headline": df["Article_title"].astype("string").str.strip(),
            "source": df["Publisher"].astype("string").fillna("unknown"),
            "url": df["Url"].astype("string").fillna(""),
            "published_at": published,
            "ticker": df["Stock_symbol"].astype("string").str.strip().str.upper(),
        }
    )
    return out


def stream_extract_universe(
    tickers: Sequence[str],
    url: str = FNSPID_NASDAQ_URL,
    out_path: str | Path = DEFAULT_PARQUET_PATH,
    start: str | None = "2005-01-01",
    end: str | None = "2023-12-31",
    chunksize: int = 200_000,
    progress_every: int = 5,
    max_retries: int = 3,
) -> dict:
    """Stream the remote CSV and keep only our universe -- never storing it.

    The 23 GB file is read straight off the HTTP response in chunks: rows are
    filtered to `tickers` and [start, end] as they arrive, so peak memory is
    proportional to `chunksize` plus the (small) filtered result, and nothing
    but the filtered parquet ever touches disk.

    `usecols` drops the article bodies and the four summary columns, which are
    the bulk of the file's weight.
    """
    out_path = Path(out_path)
    wanted = {t.strip().upper() for t in tickers}
    lo = pd.Timestamp(start, tz="UTC") if start else None
    hi = pd.Timestamp(end, tz="UTC") if end else None

    kept: list[pd.DataFrame] = []
    rows_scanned = 0
    rows_matched = 0
    rows_invalid_ts = 0
    rows_out_of_range = 0

    t0 = time.time()
    raw = _ResumableHTTPReader(url, max_retries=max_retries)
    with io.TextIOWrapper(
        io.BufferedReader(raw, buffer_size=1 << 22), encoding="utf-8", errors="replace"
    ) as text:
        reader = pd.read_csv(
            text,
            usecols=USED_COLUMNS,
            chunksize=chunksize,
            dtype=str,
            low_memory=False,
        )
        for i, chunk in enumerate(reader):
            rows_scanned += len(chunk)
            sym = chunk["Stock_symbol"].astype("string").str.strip().str.upper()
            hit = chunk[sym.isin(wanted)]
            if len(hit):
                rows_matched += len(hit)
                norm = _normalise(hit)
                rows_invalid_ts += int(norm["published_at"].isna().sum())
                norm = norm[norm["published_at"].notna()]
                before = len(norm)
                if lo is not None:
                    norm = norm[norm["published_at"] >= lo]
                if hi is not None:
                    norm = norm[norm["published_at"] <= hi]
                rows_out_of_range += before - len(norm)
                if len(norm):
                    kept.append(norm)
            if progress_every and i % progress_every == 0:
                n = sum(len(k) for k in kept)
                pct = 100 * raw.count / raw.total if raw.total else 0
                rate = raw.count / 1e6 / max(time.time() - t0, 1e-9)
                print(
                    f"     {raw.count / 1e9:5.2f} GB ({pct:4.1f}%) "
                    f"{rate:5.1f} MB/s  scanned {rows_scanned:>10,}  kept {n:>8,}",
                    flush=True,
                )
    bytes_read = raw.count
    reconnects = raw.reconnects

    if not kept:
        raise ValueError(f"no FNSPID rows matched {sorted(wanted)}")

    out = pd.concat(kept, ignore_index=True)
    out = out[out["headline"].notna() & (out["headline"].str.len() > 0)]
    out["article_id"] = [
        hashlib.sha1(f"{u}|{h}|{t}".encode()).hexdigest()[:16]
        for u, h, t in zip(out["url"], out["headline"], out["ticker"], strict=True)
    ]
    before_dedupe = len(out)
    out = out.drop_duplicates(subset=["article_id"]).reset_index(drop=True)
    out = out[NEWS_SCHEMA].sort_values(["ticker", "published_at"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    return {
        "source_url": url,
        "bytes_streamed": bytes_read,
        "stream_reconnects": reconnects,
        "rows_scanned": rows_scanned,
        "rows_matched": rows_matched,
        "rows_invalid_timestamp": rows_invalid_ts,
        "rows_outside_date_range": rows_out_of_range,
        "duplicates_removed": before_dedupe - len(out),
        "rows_written": len(out),
        "out_path": str(out_path),
        "bytes_written": out_path.stat().st_size,
        "earliest": out["published_at"].min(),
        "latest": out["published_at"].max(),
    }


class _ResumableHTTPReader(io.RawIOBase):
    """A byte-exact resumable HTTP reader.

    Streaming 23 GB over one connection reliably trips ConnectionResetError.
    Restarting the whole stream wastes everything read so far, and resuming at
    a line boundary is unsafe because FNSPID article bodies contain newlines
    inside quoted fields.

    This tracks the absolute byte offset and, on failure, reopens with
    `Range: bytes=<pos>-`. The consumer sees one uninterrupted byte stream, so
    the CSV parser never observes the seam.
    """

    def __init__(self, url: str, max_retries: int = 8, timeout: int = 180):
        self.url = url
        self.max_retries = max_retries
        self.timeout = timeout
        self.count = 0
        self.total = 0
        self.reconnects = 0
        self._resp = None
        self._ctx = ssl.create_default_context()
        self._open()

    def _open(self, attempts: int = 12) -> None:
        """(Re)open the stream at the current offset.

        Reconnecting is itself retried: a transient DNS failure
        (`getaddrinfo failed`) must not end a multi-hour download. An earlier
        version called this once from the read handler, so a momentary outage
        escaped the retry loop and killed the run at 7 GB.
        """
        headers = {"User-Agent": "ai-market-intelligence/0.1"}
        if self.count:
            headers["Range"] = f"bytes={self.count}-"
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                req = urllib.request.Request(self.url, headers=headers)
                self._resp = urllib.request.urlopen(
                    req, timeout=self.timeout, context=self._ctx
                )
                length = int(self._resp.headers.get("Content-Length", 0))
                if not self.total:
                    self.total = length
                return
            except Exception as exc:
                last = exc
                wait = min(2**attempt, 60)
                print(
                    f"[WARN] reconnect at {self.count / 1e9:.2f} GB failed "
                    f"({type(exc).__name__}); retrying in {wait}s",
                    flush=True,
                )
                time.sleep(wait)
        raise RuntimeError(f"could not reopen stream at byte {self.count}") from last

    def readable(self) -> bool:
        return True

    def readinto(self, buf) -> int:
        for attempt in range(self.max_retries):
            try:
                block = self._resp.read(len(buf))
                if not block:
                    return 0
                buf[: len(block)] = block
                self.count += len(block)
                return len(block)
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise
                self.reconnects += 1
                print(
                    f"[WARN] stream reset at {self.count / 1e9:.2f} GB "
                    f"({type(exc).__name__}); resuming",
                    flush=True,
                )
                time.sleep(min(2**attempt, 30))
                try:
                    self._resp.close()
                except Exception:
                    pass
                self._open()
        return 0

    def close(self) -> None:
        try:
            if self._resp is not None:
                self._resp.close()
        finally:
            super().close()


class _CountingReader:
    """Wraps a stream so we can report progress in bytes consumed."""

    def __init__(self, raw):
        self._raw = raw
        self.count = 0

    def read(self, size=-1):
        block = self._raw.read(size)
        self.count += len(block)
        return block

    def readinto(self, b):
        block = self.read(len(b))
        b[: len(block)] = block
        return len(block)

    def readable(self):
        return True

    def writable(self):
        return False

    def seekable(self):
        return False

    def close(self):
        return None

    @property
    def closed(self):
        return getattr(self._raw, "closed", False)


class FnspidNewsSource:
    """NewsSource over the locally extracted FNSPID parquet."""

    name = "fnspid"

    def __init__(self, path: str | Path = DEFAULT_PARQUET_PATH):
        self.path = Path(path)

    def available(self) -> bool:
        return self.path.exists()

    def fetch(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Return this ticker's articles within [start, end].

        Attribution comes solely from FNSPID's Stock_symbol -- no title or
        organization substring matching. Timestamps are the dataset's own
        publication times, never synthesised.
        """
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found; run extract_universe() to build it"
            )
        df = pd.read_parquet(self.path)
        df = df[df["ticker"] == ticker.strip().upper()].copy()
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
        if start is not None:
            df = df[df["published_at"] >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            df = df[df["published_at"] <= pd.Timestamp(end, tz="UTC")]
        return df[NEWS_SCHEMA].sort_values("published_at").reset_index(drop=True)

    def tickers(self) -> list[str]:
        if not self.path.exists():
            return []
        return sorted(pd.read_parquet(self.path, columns=["ticker"])["ticker"].unique())


def summarize(parquet_path: str | Path = DEFAULT_PARQUET_PATH) -> pd.DataFrame:
    """Per-ticker coverage of the extracted artifact."""
    df = pd.read_parquet(parquet_path)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    g = df.groupby("ticker")["published_at"]
    return pd.DataFrame(
        {
            "articles": g.size(),
            "earliest": g.min().dt.date,
            "latest": g.max().dt.date,
            "distinct_days": g.apply(lambda s: s.dt.date.nunique()),
        }
    ).sort_values("articles", ascending=False)


def iter_chunks(paths: Iterable[str | Path]) -> Iterable[pd.DataFrame]:  # pragma: no cover
    """Reserved for multi-file FNSPID variants."""
    for p in paths:
        yield pd.read_parquet(p)
