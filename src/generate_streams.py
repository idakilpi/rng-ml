import os
import random
import secrets
from pathlib import Path
import numpy as np

N_STREAMS = 20
STREAM_SIZE = 2 * 1024 * 1024

OUT = Path("data/raw")


def write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def mt(seed: int, n: int) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.getrandbits(8) for _ in range(n))


def pcg(seed: int, n: int) -> bytes:
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.bytes(n)


def lcg(seed: int, n: int) -> bytes:
    # Park–Miller minimal standard LCG
    # X_(n+1) = (16807 * X_n) mod 2147483647
    a = 16807
    m = 2147483647

    if not (1 <= seed < m):
        raise ValueError(f"seed must be in range [1, {m - 1}]")

    state = seed
    out = bytearray()

    while len(out) < n:
        state = (a * state) % m
        out.extend(state.to_bytes(4, byteorder="little", signed=False))

    return bytes(out[:n])


for i in range(N_STREAMS):
    write(OUT / "lcg" / f"stream_{i:03d}.bin", lcg(10000 + i, STREAM_SIZE))
    write(OUT / "mt" / f"stream_{i:03d}.bin", mt(20000 + i, STREAM_SIZE))
    write(OUT / "pcg64" / f"stream_{i:03d}.bin", pcg(30000 + i, STREAM_SIZE))
    write(OUT / "urandom" / f"stream_{i:03d}.bin", os.urandom(STREAM_SIZE))
    write(OUT / "secrets" / f"stream_{i:03d}.bin", secrets.token_bytes(STREAM_SIZE))