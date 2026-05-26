---
name: textbook-ocr
description: >-
  Batch OCR scanned textbook images using local Ollama GLM-OCR, clean OCR noise,
  format Markdown, and merge pages into chapter files. Use when the user has
  scanned textbook pages, textbook OCR, batch OCR, image-to-Markdown, Chinese
  textbook digitization, or chapter-based OCR cleanup needs.
metadata:
  openclaw:
    emoji: "📚"
    homepage: https://github.com/xiangjiawei1234-prog/textbook-ocr
---

# Textbook OCR

Use this skill to convert scanned textbook page images into cleaned, chapter-based
Markdown with a local Ollama GLM-OCR model.

## Core Rules For Agents

- Do not stream, tail, or repeatedly read page OCR output while OCR is running.
- Start the pipeline once, then wait for the process to finish.
- Use quiet OCR by default. Detailed per-page status belongs in local log files,
  not in the model context.
- Do not load full-book OCR text into chat to decide chapter ranges. Generate a
  compact page index or chapter plan JSON and inspect only that small file.
- Preserve the large-model cleanup path when local regex/Markdown formatting is
  not good enough. Use bounded chunks, never the full book at once.
- Prefer one end-to-end `pipeline.py` run. The separate script names mentioned by
  older versions of this skill no longer exist.

## Prerequisites

- Python 3 with `ollama` and `Pillow`
- Ollama running locally
- GLM-OCR model available in Ollama, default model name: `glm-ocr`
- Scanned images named with zero-padded numbers, for example `001.jpg`, `002.png`

Install Python dependencies:

```bash
pip install -r scripts/requirements.txt
```

## Security And Transparency

- No cloud API key is required.
- OCR requests go to the local Ollama server configured on the machine running
  the command.
- The skill reads only optional `TEXTBOOK_OCR_*` tuning variables.
- Do not commit OCR outputs, `.env` files, logs, temporary work directories, or
  source textbook scans.

## Recommended End-To-End Run

If the user already has chapter ranges:

```bash
python scripts/pipeline.py \
  --image-dir "/path/to/images" \
  --work-dir "/path/to/ocr_temp" \
  --output-dir "/path/to/chapters" \
  --chapters "/path/to/chapters.json" \
  --workers 2
```

The chapter JSON format is:

```json
[
  ["01_intro.md", "Introduction", 20, 34],
  ["02_environment_health.md", "Environment and Health", 35, 87]
]
```

Each row is:

```text
[output_filename, chapter_title, start_page, end_page]
```

## OCR Speed

OCR is parallelized with `--workers`.

- Start with `--workers 2`.
- Try `--workers 3` or `--workers 4` only if local GPU/CPU memory is stable.
- The environment variable `TEXTBOOK_OCR_WORKERS` can set the default.
- Use `--model`, `--max-dim`, `--align`, `--num-ctx`, and `--prompt` to tune
  OCR without editing `scripts/pipeline.py`.
- Existing page `.txt` files longer than 20 characters are skipped unless
  `--force-ocr` is used, so interrupted runs can resume.

Example:

```bash
python scripts/pipeline.py \
  --image-dir "/path/to/images" \
  --work-dir "/path/to/ocr_temp" \
  --output-dir "/path/to/chapters" \
  --chapters "/path/to/chapters.json" \
  --workers 4
```

Dense tables or unstable pages can be retried with smaller images and stronger
alignment:

```bash
python scripts/pipeline.py \
  --image-dir "/path/to/images" \
  --work-dir "/path/to/ocr_temp" \
  --force-ocr \
  --workers 1 \
  --max-dim 800 \
  --align 32 \
  --num-ctx 8192
```

## Quiet OCR And Token Control

The default OCR mode prints only sparse progress lines. It writes detailed status
to:

- `<work-dir>/ocr.log`
- `<work-dir>/ocr_progress.json`

Only use `--verbose` when a human explicitly wants per-page console output.

Useful options:

```bash
--progress-every 20
--log-file "/path/to/ocr.log"
--progress-file "/path/to/ocr_progress.json"
```

Agent behavior:

1. Launch the command.
2. Wait for completion.
3. Read only the final summary, the log tail, or the progress JSON if needed.
4. Never paste full OCR page text into the chat unless the user asks for a
   specific page.

## Large-Model Cleanup And Formatting

Local cleanup is fast and deterministic, but it may miss textbook structure. For
better Markdown quality, use the LLM cleanup path.

Step 1: OCR only and prepare bounded LLM chunks:

```bash
python scripts/pipeline.py \
  --image-dir "/path/to/images" \
  --work-dir "/path/to/ocr_temp" \
  --prepare-llm-cleanup \
  --llm-dir "/path/to/llm_cleanup" \
  --workers 2
```

This writes:

- `/path/to/llm_cleanup/input/chunk_001.md`
- `/path/to/llm_cleanup/input/chunk_002.md`
- `/path/to/llm_cleanup/manifest.json`
- `/path/to/llm_cleanup/cleaned/`

Step 2: Process each `input/chunk_*.md` with the large model and write the
cleaned result to the matching file in `cleaned/`.

Agent rules for each chunk:

- Read only one chunk at a time.
- Preserve every `<!-- page: NNN -->` marker exactly.
- Do not summarize or omit content.
- Clean OCR errors only when obvious.
- Improve paragraphs, heading hierarchy, lists, formulas, and table-like text.
- Write the result to the matching `cleaned/chunk_*.md` file.

Step 3: Apply the cleaned chunks and merge chapters:

```bash
python scripts/pipeline.py \
  --skip-ocr \
  --work-dir "/path/to/ocr_temp" \
  --apply-llm-cleanup "/path/to/llm_cleanup" \
  --output-dir "/path/to/chapters" \
  --chapters "/path/to/chapters.json"
```

When `--apply-llm-cleanup` is used, local regex cleanup and local Markdown
formatting are skipped by default so the large-model output is not damaged. Use
`--local-after-llm` only when explicitly desired.

To prepare larger or smaller chunks:

```bash
--llm-chunk-chars 8000
--llm-chunk-chars 16000
```

Use smaller chunks for dense pages, tables, or fragile OCR.

## Unknown Chapter Boundaries

When chapter ranges are unknown, first run OCR and cleanup, then generate compact
review files:

```bash
python scripts/pipeline.py \
  --skip-ocr \
  --work-dir "/path/to/ocr_temp" \
  --detect-only \
  --page-index-out "/path/to/page_index.json" \
  --chapter-plan-out "/path/to/chapters.detected.json"
```

Use `page_index.json` to review only:

- page number
- character count
- first few non-empty lines
- last few non-empty lines

Use `chapters.detected.json` as the merge plan after checking boundaries:

```bash
python scripts/pipeline.py \
  --skip-ocr \
  --work-dir "/path/to/ocr_temp" \
  --output-dir "/path/to/chapters" \
  --chapters "/path/to/chapters.detected.json"
```

For a fully automatic pass:

```bash
python scripts/pipeline.py \
  --image-dir "/path/to/images" \
  --work-dir "/path/to/ocr_temp" \
  --output-dir "/path/to/chapters" \
  --auto-chapters \
  --workers 2
```

## Pipeline Stages

1. OCR
   - Reads numbered `.jpg`, `.jpeg`, `.png`, and `.webp` scans.
   - Downscales large images to `--max-dim` / `TEXTBOOK_OCR_MAX_DIM`.
   - Aligns resized dimensions to `--align` / `TEXTBOOK_OCR_ALIGN` for GGML
     compatibility.
   - Sends pages to Ollama GLM-OCR.
   - Saves one text file per page.
   - For dense numeric tables, lower `--max-dim` to 800 and use `--align 32`.
     These pages often need a second targeted pass.

2. Clean
   - Removes watermarks, page numbers, repeated page headers, diagram fragments,
     and common LaTeX artifacts.
   - Converts common LaTeX commands into plain Unicode where possible.
   - Can be replaced by the large-model cleanup path.

3. Format
   - Applies Markdown heading hierarchy.
   - Normalizes blank lines.
   - Preserves numbered lists.
   - Can be replaced by the large-model formatting path.

4. Detect or Merge
   - With `--chapters`, merges pages into chapter Markdown files.
   - With `--chapter-plan-out`, writes a compact detected chapter plan.
   - With `--page-index-out`, writes a small review index to avoid context
     overflow.

## Common Issues

**OCR is too slow**

Increase `--workers` gradually. If Ollama crashes, reduce workers or lower
`--max-dim`.

**The agent is wasting tokens during OCR**

Run without `--verbose`; do not tail `ocr.log`; inspect only
`ocr_progress.json` or the final summary.

**Chapter sorting overflows context**

Use `--page-index-out` and `--chapter-plan-out`. Inspect the compact JSON files
instead of loading full page text.

**Local Markdown formatting is not good enough**

Use `--prepare-llm-cleanup`, clean one chunk at a time with the large model, then
use `--apply-llm-cleanup` before merging.

**Images return almost no text**

Large images may exceed the model's stable input size. Keep downscaling enabled
and ensure dimensions are aligned to 16-pixel boundaries.

**Need to rerun only failed or missing pages**

Run again without `--force-ocr`; existing valid `.txt` files will be skipped.

**GGML_ASSERT model crash on certain pages**

Some images trigger `GGML_ASSERT(a->ne[2] * 4 == b->ne[0])` or similar dimension
mismatch errors in the GLM-OCR GGUF model. This happens most often on dense
tables with many rows/columns.

**How to fix:** The default 16-pixel alignment is insufficient. Run a targeted
retry pass that uses:

- `--max-dim 800` (smaller than the default 1600)
- `--align 32` (larger alignment boundary)
- `--num-ctx 8192` in the model options
- `--workers 1` to avoid Ollama health-check races

Example retry strategy to try in order:

```python
STRATEGIES = [
    {'max_dim': 800, 'align': 32},
    {'max_dim': 960, 'align': 64},
    {'max_dim': 640, 'align': 16},
]
```

If the first strategy (`800` / `32`) works, stop; otherwise try the next. In
practice, `800` / `32` resolves nearly all GGML_ASSERT failures.

**Dense numeric tables fail OCR at default settings**

Textbook appendix pages (statistical tables like F临界值表, χ²分布表, t界值表)
often produce zero output at the pipeline's default `--max-dim 1600` /
`--align 16`. The OCR model may return `< 20` characters, causing the pipeline to skip
saving the page.

**How to fix:** Lower `--max-dim` to `800`-`1000` and use `--align 32`. A smaller
image reduces the token burden and lets the model focus on the table structure.
Also use a prompt that explicitly mentions tables:

```python
prompt = '请识别并输出图片中的所有文字和数字内容，包括表格'
```

Run a separate retry pass targeting only the missing pages, then re-merge with
`--skip-ocr`.

**Windows GBK encoding crashes**

On Windows, `print()` statements in pipeline.py or ad-hoc scripts may crash with
`UnicodeEncodeError: 'gbk' codec can't encode character` when Chinese text or
Unicode math symbols appear in output.

**How to fix:** Set the environment variable before running:

```bash
PYTHONIOENCODING=utf-8 python scripts/pipeline.py ...
```

**Ollama health-check failures under parallel load**

When using `--workers 2` or higher, Ollama may crash on complex pages, producing
errors like `health resp: dial tcp 127.0.0.1:XXXXX: connectex: No connection`.
The runner process dies and does not recover in time for the next request.

**How to fix:** For retrying failed pages, use `workers = 1`. The single-worker
mode avoids racing Ollama's health checks. Once all pages are OCR'd, the
merge step is CPU-only and can use higher parallelism.

**Auto chapter detection misses most chapters in Chinese textbooks**

`--auto-chapters` relies on detecting chapter heading patterns in formatted text.
For Chinese textbooks where chapter titles are short phrases (e.g., "绪论",
"t检验") without a keyword prefix like "Chapter", the detector often finds only
a fraction of the real chapters (e.g., 2 out of 19).

**How to fix:** Use the book's own table of contents (TOC) to build
`chapters.json`. The TOC is typically on the first 5–15 scanned pages. Steps:

1. Read the TOC pages to extract all chapter titles and their printed page
   numbers.
2. Determine the offset between printed page numbers and image file numbers
   (usually +12 for books with cover + front matter).
3. Check a few boundary pages by reading `ocr_temp/XXX.txt` first lines to
   verify the exact image-page where each chapter content begins.
4. Build `chapters.json` manually with the verified ranges.
5. Run the pipeline with `--chapters chapters.json`.
