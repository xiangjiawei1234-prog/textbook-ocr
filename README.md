# Textbook OCR Skill / 课本 OCR 技能

Batch OCR scanned textbook pages with a local Ollama GLM-OCR model, then clean,
format, and merge the results into chapter-based Markdown files.

使用本地 Ollama GLM-OCR 模型批量识别课本扫描页，并自动完成 OCR 文本清理、
Markdown 格式化和按章节合并。

## Features / 功能

- Local-first OCR through Ollama; no cloud API key is required.
- Batch processing for numbered page images such as `001.jpg`, `002.png`, and
  `003.webp`.
- Sparse console output with detailed progress written to local log files.
- Automatic one-pass retry for pages that fail OCR, using safer retry settings
  based on the logged error.
- Optional user decision when only a few pages still fail after retry.
- Unloads the Ollama OCR model when failed pages are skipped, so GPU memory can
  be released before cleanup and chapter merging continue.
- Local cleanup, Markdown formatting, chapter detection, and chapter merging.
- Optional chunked large-model cleanup workflow for difficult OCR text.

- 通过 Ollama 在本地完成 OCR，不需要云端 API Key。
- 支持按页编号的扫描图，例如 `001.jpg`、`002.png`、`003.webp`。
- 控制台默认只输出简要进度，详细状态写入本地日志。
- 第一遍 OCR 后会读取日志，并对失败页面自动重试一遍。
- 如果重试后仍有少量失败页，会报告页码并让用户决定是否继续尝试 OCR。
- 如果用户决定不继续 OCR，会卸载 Ollama OCR 模型，释放显存，再继续后续清理和分章节。
- 支持本地文本清理、Markdown 格式化、章节检测和章节合并。
- 对复杂教材文本可使用分块的大模型清理流程。

## Install / 安装

```bash
pip install -r scripts/requirements.txt
ollama pull glm-ocr
```

Start Ollama before running the pipeline.

运行流程前请先启动 Ollama。

## Quick Start / 快速开始

```bash
python scripts/pipeline.py \
  --image-dir "/path/to/scans" \
  --work-dir "/path/to/ocr_temp" \
  --output-dir "/path/to/chapters" \
  --chapters "references/example-chapters.json" \
  --workers 2
```

Input images should be numbered by page.

输入图片需要按页码命名。

```text
001.jpg
002.png
003.webp
```

The chapter JSON format is:

章节 JSON 格式如下：

```json
[
  ["01_intro.md", "Introduction", 20, 34],
  ["02_environment_health.md", "Environment and Health", 35, 87]
]
```

Each row is:

每一行表示：

```text
[output_filename, chapter_title, start_page, end_page]
```

## Failed Page Retry / 失败页重试

After the first OCR pass, the pipeline reads `<work-dir>/ocr.log` and finds
pages whose latest status is `FAIL`. These pages are retried once automatically.

第一遍 OCR 完成后，脚本会读取 `<work-dir>/ocr.log`，找出最新状态仍为 `FAIL`
的页面，并自动只重试这些页面一次。

Retry behavior:

重试策略：

- GGML, tensor, or dimension errors use smaller aligned images.
- Ollama health or connection errors retry with one worker.
- Other failures use conservative smaller-image retry settings.

- 遇到 GGML、tensor、dimension 等错误时，使用更小且对齐后的图片重试。
- 遇到 Ollama health 或连接错误时，改用单 worker 串行重试。
- 其他错误使用更保守的降尺寸重试设置。

Useful options:

常用参数：

```bash
--no-auto-ocr-retry
--failed-ocr-threshold 10
--failed-ocr-action ask
--failed-ocr-action continue
--failed-ocr-action retry
--failed-ocr-action abort
--keep-ollama-on-ocr-skip
```

If the user decides not to continue OCR for remaining failed pages, the pipeline
will run `ollama stop <model>` unless `--keep-ollama-on-ocr-skip` is set.

如果用户决定不再继续 OCR 剩余失败页，脚本会执行 `ollama stop <model>` 来释放
模型占用的显存，除非指定了 `--keep-ollama-on-ocr-skip`。

## Configuration / 配置

The pipeline reads these optional environment variables:

脚本支持以下可选环境变量：

| Variable / 变量 | Default / 默认值 | Purpose / 用途 |
| --- | --- | --- |
| `TEXTBOOK_OCR_MODEL` | `glm-ocr` | Ollama model name / Ollama 模型名 |
| `TEXTBOOK_OCR_WORKERS` | `2` | Parallel OCR requests / 并行 OCR 请求数 |
| `TEXTBOOK_OCR_MAX_DIM` | `1600` | Maximum image dimension before OCR / OCR 前图片最大边长 |
| `TEXTBOOK_OCR_ALIGN` | `16` | Resize alignment boundary / 图片缩放对齐边界 |
| `TEXTBOOK_OCR_NUM_CTX` | unset / 未设置 | Optional Ollama context size / 可选 Ollama 上下文长度 |
| `TEXTBOOK_OCR_PROMPT` | Chinese OCR prompt / 中文 OCR 提示词 | Prompt sent to GLM-OCR / 发给 GLM-OCR 的提示词 |
| `TEXTBOOK_OCR_FAIL_PROMPT_THRESHOLD` | `10` | Ask only when remaining failures are few / 失败页较少时才询问 |
| `TEXTBOOK_OCR_FAILED_ACTION` | `ask` | Default action for unresolved failures / 未解决失败页的默认动作 |

Equivalent CLI flags are available for the main tuning values.

主要调优项也可以通过命令行参数设置。

```bash
python scripts/pipeline.py --help
```

## Security / 安全说明

- No cloud API key is required.
- Images are sent only to the local Ollama server configured on the machine
  running the command.
- Do not commit OCR outputs, `.env` files, logs, temporary work directories, or
  source textbook scans.

- 不需要云端 API Key。
- 图片只会发送到当前机器配置的本地 Ollama 服务。
- 不要提交 OCR 输出、`.env` 文件、日志、临时工作目录或教材原始扫描图。

## Repository Contents / 仓库内容

Only source files, skill instructions, examples, and license files should be
committed:

仓库只应提交源码、技能说明、示例和许可证文件：

```text
SKILL.md
README.md
LICENSE
scripts/pipeline.py
scripts/requirements.txt
references/example-chapters.json
```

Generated OCR text, logs, temporary folders, virtual environments, and source
scans should stay local.

生成的 OCR 文本、日志、临时目录、虚拟环境和原始扫描图应保留在本地。

## License / 许可证

Apache License 2.0. See [LICENSE](LICENSE).

Apache License 2.0。详见 [LICENSE](LICENSE)。
