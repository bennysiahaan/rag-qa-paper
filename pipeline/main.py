from pathlib import Path

import yaml

from pipeline.custom_types import parse_source
from pipeline.ingestion.reader import load_from_sources

def main():
    manifest_path = Path(__file__).parent.parent / "manifest.yaml"
    with open(str(manifest_path), "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
        f.close()
    
    sources = []
    for entry in manifest["sources"]:
        source = parse_source(entry)
        sources.append(source)
    
    load_from_sources(sources)

if __name__ == "__main__":
    main()