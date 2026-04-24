# Local RAG Question Answering for Papers
A RAG for question answering task that runs on local machine. It ingests and understands `.pdf` files as context before providing answers.

## Requirements
- Python 3.11+.
- Docker.
- Ollama. Go to [Ollama's website](https://ollama.com/) to set up Ollama.

## Running the RAG
- Pull the desired model for generation via:
  ```bash
  $ ollama pull <model_name>
  ```
- Install required packages:
  ```bash
  $ pip install -r requirements_dev.txt
  ```
- Spin up the qdrant server:
  ```bash
  $ docker compose up -d
  ```
- Run by providing the directory path to the desired `.pdf` files to ingest.
  ```bash
  $ python main.py --preload <path_to_the_directory>
  ```

## Command arguments
| Argument | Description |
|----------|-------------|
|`--preload` | Provide the directory path to `.pdf` files to ingest. |
|`--cleanup_on_init` | Remove all document chunks in Qdrant storage on startup. |
|`--top_k` | The number of documents to retrieve as context. |
|`--prefetch_limit` | The number of documents to retrieve from the storage. |