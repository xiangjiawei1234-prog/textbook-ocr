# Textbook OCR Skill

Batch OCR scanned textbook pages with a local Ollama GLM-OCR model, then clean,
format, and merge the results into chapter-based Markdown files.

## Install

```bash
pip install -r scripts/requirements.txt
ollama pull glm-ocr
```

Start Ollama before running the pipeline.

## Quick Start

```bash
python scripts/pipeline.py \
  --image-dir "/path/to/scans" \
  --work-dir "/path/to/ocr_temp" \
  --output-dir "/path/to/chapters" \
  --chapters "references/example-chapters.json" \
  --workers 2
```

Input images should be numbered by page, for example `001.jpg`, `002.png`, or
`003.webp`.

## Configuration

The pipeline is local-first and reads these optional environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `TEXTBOOK_OCR_MODEL` | `glm-ocr` | Ollama model name |
| `TEXTBOOK_OCR_WORKERS` | `2` | Parallel OCR requests |
| `TEXTBOOK_OCR_MAX_DIM` | `1600` | Maximum image dimension before OCR |
| `TEXTBOOK_OCR_ALIGN` | `16` | Resize alignment boundary |
| `TEXTBOOK_OCR_NUM_CTX` | unset | Optional Ollama context size |
| `TEXTBOOK_OCR_PROMPT` | Chinese text OCR prompt | Prompt sent to GLM-OCR |

Equivalent CLI flags are available for the main tuning values:

```bash
python scripts/pipeline.py --help
```

## Security

- No cloud API key is required.
- The script sends images only to the local Ollama server configured on the
  machine running the command.
- Do not commit OCR outputs, `.env` files, logs, or temporary work directories.

## License

Apache License 2.0. See [LICENSE](LICENSE).
