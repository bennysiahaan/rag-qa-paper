from dataclasses import dataclass, field

@dataclass
class SourceBase:
    type: str
    path: str
    glob: str

def parse_source(entry: dict) -> SourceBase:
    return SourceBase(**entry)

SUPPORTED_DOCUMENT_TYPES = ["pdf", "docx"]

@dataclass
class RawDocument:
    source_id: str
    source_uri: str
    content: bytes | str
    content_hash: str
    bucket: str = ""
    metadata: dict = field(default_factory=dict)