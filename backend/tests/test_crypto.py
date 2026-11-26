import os

import pytest

from app.gate.crypto import (DecryptionError, decrypt_file, encrypt_file, generate_data_key,
                             decode_key, encode_key)

CHUNK = 64 * 1024
ENC_CHUNK = CHUNK + 16
HEADER = 17


def roundtrip_setup(tmp_path, size=250_000, chunk=CHUNK):
    plain = os.urandom(size)
    src, enc, out = tmp_path / "p.bin", tmp_path / "e.enc", tmp_path / "o.bin"
    src.write_bytes(plain)
    key = generate_data_key()
    encrypt_file(src, enc, key, chunk_size=chunk)
    return plain, enc, out, key


def test_roundtrip_multi_chunk(tmp_path):
    plain, enc, out, key = roundtrip_setup(tmp_path)
    decrypt_file(enc, out, key)
    assert out.read_bytes() == plain


def test_roundtrip_exact_chunk_multiple_and_tiny(tmp_path):
    for size in (CHUNK, CHUNK * 3, 1):
        plain, enc, out, key = roundtrip_setup(tmp_path, size=size)
        decrypt_file(enc, out, key)
        assert out.read_bytes() == plain


def test_ciphertext_has_no_plaintext(tmp_path):
    marker = b"PATIENT-SECRET-MARKER" * 50
    src, enc = tmp_path / "p.bin", tmp_path / "e.enc"
    src.write_bytes(marker)
    encrypt_file(src, enc, generate_data_key())
    assert b"PATIENT-SECRET-MARKER" not in enc.read_bytes()


def test_bit_flip_detected_and_no_output_left(tmp_path):
    _, enc, out, key = roundtrip_setup(tmp_path)
    data = bytearray(enc.read_bytes())
    data[100_000] ^= 1
    enc.write_bytes(bytes(data))
    with pytest.raises(DecryptionError):
        decrypt_file(enc, out, key)
    assert not out.exists()


def test_truncation_at_chunk_boundary_detected(tmp_path):
    _, enc, out, key = roundtrip_setup(tmp_path, size=CHUNK * 3)
    data = enc.read_bytes()
    enc.write_bytes(data[: HEADER + 2 * ENC_CHUNK])  # drop the last chunk
    with pytest.raises(DecryptionError):
        decrypt_file(enc, out, key)


def test_chunk_reorder_detected(tmp_path):
    _, enc, out, key = roundtrip_setup(tmp_path, size=CHUNK * 3)
    data = enc.read_bytes()
    h = data[:HEADER]
    c0 = data[HEADER:HEADER + ENC_CHUNK]
    c1 = data[HEADER + ENC_CHUNK:HEADER + 2 * ENC_CHUNK]
    c2 = data[HEADER + 2 * ENC_CHUNK:]
    enc.write_bytes(h + c1 + c0 + c2)
    with pytest.raises(DecryptionError):
        decrypt_file(enc, out, key)


def test_wrong_key_rejected(tmp_path):
    _, enc, out, _ = roundtrip_setup(tmp_path)
    with pytest.raises(DecryptionError):
        decrypt_file(enc, out, generate_data_key())


def test_key_encode_roundtrip_and_malformed_rejected():
    dk = generate_data_key()
    assert decode_key(encode_key(dk)) == dk
    with pytest.raises(DecryptionError):
        decode_key(encode_key(b"short"))
