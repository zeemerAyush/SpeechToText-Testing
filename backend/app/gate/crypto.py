"""Envelope encryption for recordings (AES-256-GCM, chunked).

File layout of ``rec_<id>.enc``:
    MAGIC(4) | version(1) | chunk_size(4, big endian) | nonce_prefix(8) | chunks...
Each chunk is AES-GCM(ciphertext + 16-byte tag) with
    nonce = nonce_prefix || counter(4, big endian)
    aad   = header || counter(4) || final_flag(1)
The counter and the final flag in the AAD stop chunk reordering and truncation.
Each recording gets its own random data key, stored (base64) in its MongoDB record.
(Test phase: there is no master key. Add key wrapping back before real patient data.)
"""
from __future__ import annotations

import base64
import os
import struct
from pathlib import Path
from typing import BinaryIO

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"STT1"
VERSION = 1
CHUNK_SIZE = 1024 * 1024
TAG_LEN = 16
HEADER_LEN = 4 + 1 + 4 + 8


class DecryptionError(Exception):
    """Raised when data is corrupted, truncated, tampered with, or the key is wrong."""


def generate_data_key() -> bytes:
    return AESGCM.generate_key(bit_length=256)


def encode_key(data_key: bytes) -> str:
    return base64.b64encode(data_key).decode()


def decode_key(encoded: str) -> bytes:
    key = base64.b64decode(encoded)
    if len(key) != 32:
        raise DecryptionError("Stored data key is malformed.")
    return key


def _nonce(prefix: bytes, counter: int) -> bytes:
    return prefix + struct.pack(">I", counter)


def _aad(header: bytes, counter: int, final: bool) -> bytes:
    return header + struct.pack(">IB", counter, 1 if final else 0)


def _read_full(fh: BinaryIO, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        part = fh.read(n - len(buf))
        if not part:
            break
        buf += part
    return buf


def encrypt_file(src: Path, dst: Path, data_key: bytes, chunk_size: int = CHUNK_SIZE) -> None:
    aes = AESGCM(data_key)
    prefix = os.urandom(8)
    header = MAGIC + bytes([VERSION]) + struct.pack(">I", chunk_size) + prefix
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_suffix(dst.suffix + ".part")
    try:
        with open(src, "rb") as fin, open(tmp_dst, "wb") as fout:
            fout.write(header)
            counter = 0
            chunk = _read_full(fin, chunk_size)
            while True:
                nxt = _read_full(fin, chunk_size) if chunk else b""
                final = not nxt
                fout.write(aes.encrypt(_nonce(prefix, counter), chunk, _aad(header, counter, final)))
                if final:
                    break
                chunk, counter = nxt, counter + 1
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(tmp_dst, dst)
    except BaseException:
        tmp_dst.unlink(missing_ok=True)
        raise


def decrypt_file(src: Path, dst: Path, data_key: bytes) -> None:
    aes = AESGCM(data_key)
    try:
        with open(src, "rb") as fin, open(dst, "wb") as fout:
            header = _read_full(fin, HEADER_LEN)
            if len(header) != HEADER_LEN or header[:4] != MAGIC or header[4] != VERSION:
                raise DecryptionError("Not a valid encrypted recording.")
            (chunk_size,) = struct.unpack(">I", header[5:9])
            prefix = header[9:17]
            enc_size = chunk_size + TAG_LEN
            counter = 0
            block = _read_full(fin, enc_size)
            if not block:
                raise DecryptionError("Encrypted recording is empty or truncated.")
            while True:
                nxt = _read_full(fin, enc_size)
                final = not nxt
                try:
                    fout.write(aes.decrypt(_nonce(prefix, counter), block, _aad(header, counter, final)))
                except InvalidTag as exc:
                    raise DecryptionError("Authentication failed: data corrupted, truncated or tampered.") from exc
                if final:
                    break
                block, counter = nxt, counter + 1
    except BaseException:
        dst.unlink(missing_ok=True)
        raise
