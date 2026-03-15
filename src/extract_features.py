from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

WINDOW = 64 * 1024  
RAW_ROOT = Path("data/raw")
OUT_FILE = Path("data/features/features.parquet")

RNGS = ["lcg", "mt", "pcg64", "urandom", "secrets"]

def iter_windows(b: bytes, size: int):
    for i in range(0, len(b), size):
        w = b[i:i+size]
        if len(w) == size:
            yield w

def label_3class(rng: str) -> str:
    if rng == "lcg":
        return "weak"
    if rng in {"mt", "pcg64"}:
        return "prng"
    return "csprng" 

def shannon_entropy_bytes(data: bytes) -> float:
    x = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(x, minlength=256).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def chi_square_uniform(data: bytes) -> float:
    x = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(x, minlength=256).astype(np.float64)
    expected = np.full(256, len(x)/256.0, dtype=np.float64)
    return float(((counts - expected) ** 2 / expected).sum())

def bit_balance(data: bytes) -> float:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return float(bits.mean())

def bit_autocorr_lag1(data: bytes) -> float:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.float64)
    a, b = bits[:-1], bits[1:]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def byte_autocorr_lag1(data: bytes) -> float:
    x = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
    a, b = x[:-1], x[1:]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def spectral_flatness(data: bytes) -> float:
    
    x = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x)) + 1e-12
    geo = np.exp(np.mean(np.log(spec)))
    ar = np.mean(spec)
    return float(geo / ar)

def spectral_peak_ratio(data: bytes) -> float:
    x = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x)) + 1e-12
    return float(spec.max() / spec.mean())

def main() -> None:
    rows = []
    for rng in RNGS:
        files = sorted((RAW_ROOT / rng).glob("stream_*.bin"))
        for f in tqdm(files, desc=f"features:{rng}"):
            stream_id = f"{rng}/{f.stem}" 
            data = f.read_bytes()
            for w_i, w in enumerate(iter_windows(data, WINDOW)):
                rows.append({
                    "rng": rng,
                    "label3": label_3class(rng),
                    "stream_id": stream_id,
                    "window_idx": w_i,
                    "entropy": shannon_entropy_bytes(w),
                    "chi2": chi_square_uniform(w),
                    "bit_balance": bit_balance(w),
                    "byte_autocorr1": byte_autocorr_lag1(w),
                    "bit_autocorr1": bit_autocorr_lag1(w),
                    "spec_flatness": spectral_flatness(w),
                    "spec_peak_ratio": spectral_peak_ratio(w),
                })

    df = pd.DataFrame(rows)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)
    print(f"Wrote {len(df)} rows -> {OUT_FILE}")

if __name__ == "__main__":
    main()
