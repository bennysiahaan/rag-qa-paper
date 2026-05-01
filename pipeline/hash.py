import hashlib

def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def hash_text(text: str) -> str:
    normalized_text = " ".join(text.strip().split())
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()