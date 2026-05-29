#!/usr/bin/env python3
"""
Textbook OCR Pipeline — 课本扫描图片 → 清洁 Markdown 章节
Usage: python pipeline.py --image-dir 图片/ --output-dir 输出/ [--chapters chapters.json]
"""

import sys, io, re, os, time, json, base64, argparse, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ============================================================
# 0. Configuration
# ============================================================

def int_env(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


MAX_DIM = int_env('TEXTBOOK_OCR_MAX_DIM', 1600)
IMAGE_ALIGN = int_env('TEXTBOOK_OCR_ALIGN', 16)
OCR_MODEL = os.environ.get('TEXTBOOK_OCR_MODEL', 'glm-ocr')
OCR_PROMPT = os.environ.get('TEXTBOOK_OCR_PROMPT', '请识别并输出图片中的所有文字内容')
OCR_NUM_CTX = int_env('TEXTBOOK_OCR_NUM_CTX', 0)
JPEG_QUALITY = 90
SLEEP_BETWEEN = 0.0
MAX_RETRIES = 3
MIN_VALID_CHARS = 20
DEFAULT_WORKERS = max(1, int(os.environ.get('TEXTBOOK_OCR_WORKERS', '2')))
DEFAULT_FAIL_PROMPT_THRESHOLD = int_env('TEXTBOOK_OCR_FAIL_PROMPT_THRESHOLD', 10)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

# ============================================================
# 1. Image preprocessing
# ============================================================

def preprocess_image(img_path):
    """Resize image to fit MAX_DIM, align dimensions, return base64."""
    from PIL import Image
    img = Image.open(img_path)
    w, h = img.size
    ratio = min(MAX_DIM / w, MAX_DIM / h)
    if ratio < 1:
        align = max(1, int(IMAGE_ALIGN))
        new_w = max(align, (int(w * ratio) // align) * align)
        new_h = max(align, (int(h * ratio) // align) * align)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()


# ============================================================
# 2. OCR via Ollama
# ============================================================

def ocr_image(img_path):
    """Send image to GLM-OCR, return recognized text."""
    import ollama
    img_b64 = preprocess_image(img_path)
    options = {'temperature': 0}
    if OCR_NUM_CTX:
        options['num_ctx'] = OCR_NUM_CTX
    for attempt in range(MAX_RETRIES):
        try:
            response = ollama.chat(
                model=OCR_MODEL,
                messages=[{'role': 'user', 'content': OCR_PROMPT, 'images': [img_b64]}],
                options=options,
            )
            return response['message']['content']
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2)
    return ''


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def append_log(log_file, message):
    if not log_file:
        return
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open('a', encoding='utf-8') as f:
        f.write(message.rstrip() + '\n')


def page_key(value):
    stem = Path(str(value)).stem
    return int(stem) if stem.isdigit() else stem.lower()


def read_ocr_failures_from_log(log_file, start_offset=0):
    """Read OCR log lines and return pages whose latest status is FAIL."""
    log_file = Path(log_file)
    if not log_file.exists():
        return {}

    status_re = re.compile(
        r'^\[\d+/\d+\]\s+(.+?\.(?:jpg|jpeg|png|webp))\s+(OK|SKIP|FAIL)\b\s*(.*)$',
        re.I,
    )
    failures = {}
    with log_file.open('rb') as f:
        if start_offset:
            f.seek(start_offset)
        text = f.read().decode('utf-8', errors='replace')
        for line in text.splitlines():
            m = status_re.match(line.strip())
            if not m:
                continue
            filename, status, detail = m.group(1), m.group(2).upper(), m.group(3).strip()
            key = page_key(filename)
            if status == 'FAIL':
                failures[key] = {'filename': filename, 'error': detail or 'unknown error'}
            else:
                failures.pop(key, None)
    return failures


def retry_settings_for_errors(failures):
    """Choose a conservative retry strategy from logged error messages."""
    error_text = ' '.join(item['error'] for item in failures.values()).lower()
    settings = {
        'workers': 1,
        'max_dim': MAX_DIM,
        'align': IMAGE_ALIGN,
        'num_ctx': OCR_NUM_CTX,
        'reason': 'single-worker retry after OCR failures',
    }

    dimension_markers = (
        'ggml_assert', 'ne[2]', 'dimension', 'tensor', 'shape', 'clip',
    )
    health_markers = (
        'health resp', 'dial tcp', 'connection refused', 'connectex',
        'server error', 'runner process',
    )

    if any(marker in error_text for marker in dimension_markers):
        settings.update({
            'max_dim': 800,
            'align': 32,
            'num_ctx': max(OCR_NUM_CTX, 8192),
            'reason': 'dimension/assert failure; retrying with smaller aligned images',
        })
    elif any(marker in error_text for marker in health_markers):
        settings.update({
            'reason': 'Ollama health/connection failure; retrying serially',
        })
    else:
        settings.update({
            'max_dim': min(MAX_DIM, 1000),
            'align': max(IMAGE_ALIGN, 32),
            'num_ctx': max(OCR_NUM_CTX, 8192),
            'reason': 'generic OCR failure; retrying with safer image settings',
        })

    return settings


def retry_failed_pages_once(image_dir, output_dir, failures, quiet=True,
                            progress_every=10, log_file=None, progress_file=None):
    """Retry only failed pages once using settings chosen from the OCR log."""
    global MAX_DIM, IMAGE_ALIGN, OCR_NUM_CTX

    if not failures:
        return {}

    settings = retry_settings_for_errors(failures)
    old_settings = (MAX_DIM, IMAGE_ALIGN, OCR_NUM_CTX)
    MAX_DIM = settings['max_dim']
    IMAGE_ALIGN = settings['align']
    OCR_NUM_CTX = settings['num_ctx']

    pages = sorted(failures)
    page_list = ', '.join(f"{p:03d}" if isinstance(p, int) else str(p) for p in pages)
    print(
        f"[OCR] Retrying failed pages once ({len(pages)}): {page_list}; "
        f"{settings['reason']}; max_dim={MAX_DIM}, align={IMAGE_ALIGN}, "
        f"num_ctx={OCR_NUM_CTX}, workers={settings['workers']}"
    )
    append_log(
        log_file,
        f"[OCR] retry start pages={page_list} reason={settings['reason']} "
        f"max_dim={MAX_DIM} align={IMAGE_ALIGN} num_ctx={OCR_NUM_CTX} "
        f"workers={settings['workers']}",
    )

    try:
        return batch_ocr(
            image_dir,
            output_dir,
            force=True,
            workers=settings['workers'],
            quiet=quiet,
            progress_every=progress_every,
            log_file=log_file,
            progress_file=progress_file,
            target_pages=pages,
            phase='ocr-retry',
        )
    finally:
        MAX_DIM, IMAGE_ALIGN, OCR_NUM_CTX = old_settings


def stop_ollama_model(log_file=None):
    """Unload the OCR model so Ollama releases GPU memory before later stages."""
    cmd = ['ollama', 'stop', OCR_MODEL]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        message = "[Ollama] Cannot stop model: ollama command was not found"
    except Exception as e:
        message = f"[Ollama] Cannot stop model: {e}"
    else:
        if result.returncode == 0:
            message = f"[Ollama] Stopped model {OCR_MODEL}"
        else:
            detail = (result.stderr or result.stdout or '').strip()
            message = f"[Ollama] Could not stop model {OCR_MODEL}: {detail}"
    print(message)
    append_log(log_file, message)


def print_failed_page_report(failures, max_pages=None):
    pages = sorted(failures)
    print(f"[OCR] Remaining failed pages after retry: {len(pages)}")
    shown = pages if max_pages is None or len(pages) <= max_pages else pages[:max_pages]
    for key in shown:
        item = failures[key]
        page = f"{key:03d}" if isinstance(key, int) else str(key)
        print(f"  page {page} ({item['filename']}): {item['error']}")
    omitted = len(pages) - len(shown)
    if omitted > 0:
        print(f"  ... {omitted} more failed pages omitted from console report")


def should_retry_after_report(failures, threshold, action):
    """Return True when the user/action asks for one more targeted retry."""
    if not failures:
        return False

    if action == 'retry':
        return True
    if action == 'abort':
        raise SystemExit(2)
    if action == 'continue':
        return False

    if len(failures) > threshold:
        print(
            f"[OCR] Remaining failures exceed prompt threshold "
            f"({len(failures)} > {threshold}); continuing without more OCR."
        )
        return False

    if not sys.stdin.isatty():
        print("[OCR] Non-interactive shell; continuing without more OCR.")
        return False

    answer = input("Retry these failed pages one more time? [y/N] ").strip().lower()
    return answer in {'y', 'yes'}


def batch_ocr(image_dir, output_dir, force=False, workers=DEFAULT_WORKERS,
              quiet=True, progress_every=10, log_file=None, progress_file=None,
              target_pages=None, phase='ocr'):
    """Run OCR on numbered .jpg files, save one .txt per page.

    Quiet mode keeps stdout small so agents do not stream every page into context.
    Detailed per-page status is written to log_file/progress_file instead.
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(workers or 1))

    images = sorted(
        [f for f in image_dir.iterdir()
         if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS and f.stem.isdigit()],
        key=lambda x: int(x.stem)
    )
    if target_pages:
        target_keys = {page_key(p) for p in target_pages}
        images = [f for f in images if page_key(f.name) in target_keys]
    total = len(images)
    print(f"[OCR] Found {total} images; workers={workers}")
    if log_file:
        append_log(log_file, f"[OCR] start phase={phase} total={total} workers={workers}")

    queued = []
    skipped = 0
    failed = 0
    done = 0

    for i, img_path in enumerate(images, 1):
        out_file = output_dir / f"{img_path.stem}.txt"

        if out_file.exists() and not force:
            existing = out_file.read_text(encoding='utf-8').strip()
            if len(existing) > MIN_VALID_CHARS:
                skipped += 1
                done += 1
                append_log(log_file, f"[{i}/{total}] {img_path.name} SKIP {len(existing)} chars")
                continue
        queued.append((i, img_path, out_file))

    def update_progress(current=None):
        if not progress_file:
            return
        write_json(progress_file, {
            'stage': phase,
            'total': total,
            'done': done,
            'queued': len(queued),
            'skipped': skipped,
            'failed': failed,
            'workers': workers,
            'current': current,
            'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        })

    update_progress()

    if not queued:
        print(f"[OCR] Done; skipped={skipped}, failed=0")
        if log_file:
            append_log(log_file, f"[OCR] done phase={phase} skipped={skipped} failed=0")
        return {
            'total': total,
            'skipped': skipped,
            'failed': 0,
            'failed_pages': {},
        }

    def run_one(item):
        i, img_path, out_file = item
        text = ocr_image(img_path)
        return i, img_path, out_file, text

    failed_pages = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(run_one, item): item for item in queued}
        for future in as_completed(future_map):
            i, img_path, out_file = future_map[future]
            try:
                _, _, _, text = future.result()
                out_file.write_text(text, encoding='utf-8')
                status = f"[{i}/{total}] {img_path.name} OK {len(text)} chars"
            except Exception as e:
                failed += 1
                failed_pages[page_key(img_path.name)] = {
                    'filename': img_path.name,
                    'error': str(e) or type(e).__name__,
                }
                status = f"[{i}/{total}] {img_path.name} FAIL {e}"

            done += 1
            append_log(log_file, status)
            update_progress(img_path.name)
            if not quiet or done == total or done % max(1, progress_every) == 0:
                print(f"[OCR] {done}/{total} done, skipped={skipped}, failed={failed}")
            if SLEEP_BETWEEN:
                time.sleep(SLEEP_BETWEEN)

    print(f"[OCR] Done; pages={total}, skipped={skipped}, failed={failed}")
    if log_file:
        append_log(log_file, f"[OCR] done phase={phase} total={total} skipped={skipped} failed={failed}")
    return {
        'total': total,
        'skipped': skipped,
        'failed': failed,
        'failed_pages': failed_pages,
    }


# ============================================================
# 3. Text cleanup
# ============================================================

def clean_text(text):
    """Remove OCR noise: watermarks, LaTeX tags, page numbers, etc."""
    lines = text.split('\n')
    result = []
    prev_empty = False

    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            if not prev_empty:
                result.append('')
                prev_empty = True
            continue
        prev_empty = False

        # Watermarks
        if s in ('公卫人', '小分队') or re.match(r'^公众号[：:]\s*公卫人小分队', s):
            continue

        # LaTeX circled numbers
        s = re.sub(r'\$\\textcircled\{\d+\}\$', '', s)

        # Page headers
        if re.match(r'^第[一二三四五六七八九十]+章\s+\S+$', s) and len(s) < 35:
            continue
        if re.match(r'^第[一二三四五六七八九十]+章$', s):
            continue

        # Diagram fragments (cluster of short lines)
        if len(s) <= 15 and not re.match(r'^[（\(][一二三四五六七八九十]+[）\)]', s) \
           and not re.match(r'^(案例|思考题|图\d|表\d)', s) \
           and not re.match(r'^[一二三四五六七八九十]+、', s) \
           and not re.match(r'^\d+\.', s):
            start = max(0, i-2)
            end = min(len(lines), i+3)
            short_count = sum(1 for j in range(start, end)
                            if lines[j].strip() and len(lines[j].strip()) <= 15
                            and not re.match(r'^[（\(][一二三四五六七八九十]+[）\)]', lines[j].strip())
                            and not re.match(r'^(案例|思考题)', lines[j].strip()))
            if short_count >= 3:
                continue

        result.append(s)

    text = '\n'.join(result)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove trailing page number
    lines = text.split('\n')
    if lines:
        last = lines[-1].strip()
        if re.match(r'^\d{1,3}$', last):
            lines = lines[:-1]
        elif last == '' and len(lines) >= 2:
            if re.match(r'^\d{1,3}$', lines[-2].strip()):
                lines = lines[:-2] + ['']
    text = '\n'.join(lines).strip()
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


def convert_latex_to_unicode(text):
    """Convert LaTeX math to Unicode equivalents."""
    # Greek letters
    greek = {
        '\\alpha': 'α', '\\beta': 'β', '\\gamma': 'γ', '\\delta': 'δ',
        '\\epsilon': 'ε', '\\zeta': 'ζ', '\\eta': 'η', '\\theta': 'θ',
        '\\iota': 'ι', '\\kappa': 'κ', '\\lambda': 'λ', '\\mu': 'μ',
        '\\nu': 'ν', '\\xi': 'ξ', '\\pi': 'π', '\\rho': 'ρ',
        '\\sigma': 'σ', '\\tau': 'τ', '\\upsilon': 'υ', '\\phi': 'φ',
        '\\chi': 'χ', '\\psi': 'ψ', '\\omega': 'ω',
        '\\Gamma': 'Γ', '\\Delta': 'Δ', '\\Theta': 'Θ', '\\Lambda': 'Λ',
        '\\Xi': 'Ξ', '\\Pi': 'Π', '\\Sigma': 'Σ', '\\Phi': 'Φ',
        '\\Psi': 'Ψ', '\\Omega': 'Ω',
    }
    for latex, uni in greek.items():
        text = text.replace(latex, uni)

    # Math symbols
    symbols = {
        '\\rightarrow': '→', '\\leftarrow': '←', '\\leftrightarrow': '↔',
        '\\Rightarrow': '⇒', '\\Leftarrow': '⇐',
        '\\geq': '≥', '\\leq': '≤', '\\neq': '≠',
        '\\approx': '≈', '\\equiv': '≡', '\\sim': '∼',
        '\\propto': '∝', '\\infty': '∞',
        '\\times': '×', '\\div': '÷', '\\pm': '±', '\\mp': '∓',
        '\\cdot': '·', '\\cdots': '⋯', '\\vdots': '⋮', '\\ddots': '⋱',
        '\\sum': 'Σ', '\\prod': 'Π', '\\int': '∫',
        '\\sqrt': '√', '\\log': 'log', '\\ln': 'ln',
        '\\max': 'max', '\\min': 'min',
        '\\partial': '∂', '\\nabla': '∇', '\\circ': '°', '\\degree': '°',
        '\\longrightarrow': '⟶', '\\longleftarrow': '⟵',
        '\\mapsto': '↦', '\\to': '→', '\\gets': '←',
        '\\uparrow': '↑', '\\downarrow': '↓',
        '\\langle': '⟨', '\\rangle': '⟩',
        '\\parallel': '∥', '\\perp': '⊥', '\\angle': '∠', '\\triangle': '△',
        '\\subset': '⊂', '\\supset': '⊃', '\\subseteq': '⊆', '\\supseteq': '⊇',
        '\\in': '∈', '\\notin': '∉', '\\ni': '∋',
        '\\forall': '∀', '\\exists': '∃', '\\emptyset': '∅',
        '\\cup': '∪', '\\cap': '∩', '\\wedge': '∧', '\\vee': '∨', '\\neg': '¬',
        '\\oplus': '⊕', '\\otimes': '⊗',
        '\\mid': '|', '\\vert': '|', '\\lvert': '|', '\\rvert': '|',
        '\\dots': '…', '\\ldots': '…', '\\quad': ' ', '\\qquad': '  ',
        '\\textgreater': '>', '\\textless': '<',
        '\\%': '%', '\\_': '_', '\\#': '#', '\\&': '&',
        '\\backslash': '\\',
        '\\rightleftharpoons': '⇌',
    }
    for latex, uni in symbols.items():
        text = text.replace(latex, uni)

    # Subscript/superscript
    sub_map = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
    }
    sup_map = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    }

    text = re.sub(r'\$_\{([^}]+)\}\$', lambda m: ''.join(sub_map.get(c, c) for c in m.group(1)), text)
    text = re.sub(r'\$\^\{([^}]+)\}\$', lambda m: ''.join(sup_map.get(c, c) for c in m.group(1)), text)
    text = re.sub(r'\$_(\w)\$', lambda m: sub_map.get(m.group(1), m.group(1)), text)
    text = re.sub(r'\$\^(\w)\$', lambda m: sup_map.get(m.group(1), m.group(1)), text)

    # \text{...}, \mathrm{...} → content
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', text)

    # \frac{a}{b} → (a)/(b)
    for _ in range(3):
        text = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', text)

    # Bracket commands
    text = re.sub(r'\\left\(?', '(', text)
    text = re.sub(r'\\right\)?', ')', text)
    text = re.sub(r'\\left\[', '[', text)
    text = re.sub(r'\\right\]', ']', text)
    text = re.sub(r'\\left\{', '{', text)
    text = re.sub(r'\\right\{', '}', text)

    # Residual commands
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = text.replace('{', '').replace('}', '')
    text = text.replace('$', '')
    text = re.sub(r' +', ' ', text)

    return text


def clean_all(result_dir):
    """Clean all OCR result files in directory."""
    result_dir = Path(result_dir)
    files = sorted(result_dir.glob("*.txt"), key=lambda x: int(x.stem))
    print(f"[Clean] Processing {len(files)} files")
    for f in files:
        text = f.read_text(encoding='utf-8').strip()
        text = clean_text(text)
        text = convert_latex_to_unicode(text)
        f.write_text(text, encoding='utf-8')
    print("[Clean] Done")


# ============================================================
# 4. Markdown formatting
# ============================================================

def format_markdown(text):
    """Apply markdown heading hierarchy to OCR text."""
    lines = text.split('\n')
    result = []
    prev_empty = False

    for line in lines:
        s = line.strip()
        if not s:
            if not prev_empty:
                result.append('')
                prev_empty = True
            continue
        prev_empty = False

        # H2: 第X章 →
        m = re.match(r'^(第[一二三四五六七八九十]+章)\s*(.*)', s)
        if m and len(s) < 40:
            ch_num = m.group(1)
            rest = m.group(2).strip()
            heading = f"## {ch_num} {rest}" if rest else f"## {ch_num}"
            result.append(heading); result.append(''); continue

        # H3: 第X节 →
        m = re.match(r'^(第[一二三四五六七八九十]+节)\s*(.*)', s)
        if m and len(s) < 50:
            sec = m.group(1); rest = m.group(2).strip()
            heading = f"### {sec} {rest}" if rest else f"### {sec}"
            result.append(heading); result.append(''); continue

        # H4: 一、... →
        m = re.match(r'^([一二三四五六七八九十]+)、(.+)', s)
        if m and len(s) < 80:
            result.append(f"#### {s}"); result.append(''); continue

        # H5: （一）... →
        m = re.match(r'^(（[一二三四五六七八九十]+）)(.+)', s)
        if m and len(s) < 80:
            result.append(f"##### {s}"); result.append(''); continue

        # Numbered list: "1. ..."
        m = re.match(r'^(\d+)\.\s+(.+)', s)
        if m:
            result.append(f"{m.group(1)}. {m.group(2)}"); continue

        result.append(s)

    text = '\n'.join(result)
    text = re.sub(r'\n{3,}', '\n\n', text)
    for lvl in ['##', '###', '####', '#####']:
        text = re.sub(r'([^\n])\n(' + lvl + r' )', r'\1\n\n\2', text)
    return text.strip()


def format_all(result_dir):
    """Apply markdown formatting to all OCR files."""
    result_dir = Path(result_dir)
    files = sorted(result_dir.glob("*.txt"), key=lambda x: int(x.stem))
    print(f"[Format] Processing {len(files)} files")
    for f in files:
        text = f.read_text(encoding='utf-8').strip()
        text = format_markdown(text)
        f.write_text(text, encoding='utf-8')
    print("[Format] Done")


# ============================================================
# 5. Chapter merging
# ============================================================

def merge_chapters(result_dir, output_dir, chapters):
    """Merge individual page files into chapter markdown files."""
    result_dir = Path(result_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, title, start, end in chapters:
        lines = [f"# {title}", ""]
        for idx in range(start, end + 1):
            f = result_dir / f"{idx:03d}.txt"
            if not f.exists():
                continue
            text = f.read_text(encoding='utf-8').strip()
            if not text:
                continue
            lines.append(text)
            lines.append("")

        out_path = output_dir / filename
        out_path.write_text('\n'.join(lines), encoding='utf-8')
        chars = sum(len(l) for l in lines)
        print(f"  {filename}: {end-start+1} pages, {chars:,} chars")

    print(f"[Merge] Done → {output_dir}")


# ============================================================
# 6. Chapter boundary detection
# ============================================================

def detect_chapters(result_dir):
    """Scan OCR results and suggest chapter page ranges."""
    result_dir = Path(result_dir)
    files = sorted(result_dir.glob("*.txt"), key=lambda x: int(x.stem))

    boundaries = []
    for f in files:
        text = f.read_text(encoding='utf-8').strip()
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        # Skip TOC pages: check for "……" dots or "目录" in first lines
        head_text = '\n'.join(lines[:5])
        if '……' in head_text or '目录' in head_text or '目 录' in head_text:
            continue
        # Check first 3 lines for chapter heading (raw OCR: "第一章\n绪 论" pattern)
        for i, line in enumerate(lines[:3]):
            m = re.match(r'^(?:#+\s*)?(第[一二三四五六七八九十]+章)(?:\s+(.*))?', line)
            if m and i <= 1:  # heading must be in first 2 non-empty lines
                ch_num = m.group(1)
                ch_title = m.group(2) or ''
                # If it's a bare "第X章", the subtitle might be on the next line
                if not ch_title and i+1 < len(lines):
                    next_line = lines[i+1]
                    if not re.match(r'^(?:#+\s*)?第', next_line) and len(next_line) < 30:
                        ch_title = next_line
                boundaries.append((int(f.stem), ch_num, ch_title))
                break

    if boundaries:
        print("[Detect] Chapter start pages:")
        for page, num, title in boundaries:
            print(f"  Page {page:03d}: {num} {title}")

    return boundaries


def get_page_bounds(result_dir):
    files = sorted(Path(result_dir).glob("*.txt"), key=lambda x: int(x.stem))
    pages = [int(f.stem) for f in files if f.stem.isdigit()]
    if not pages:
        return None, None
    return min(pages), max(pages)


def safe_filename(text):
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', '_', text)
    text = re.sub(r'\s+', '_', text).strip('._ ')
    return text[:80] or 'chapter'


def build_chapter_plan(boundaries, last_page):
    """Convert detected chapter starts into merge_chapters-compatible JSON rows."""
    chapters = []
    for idx, (start, num, title) in enumerate(boundaries, 1):
        end = boundaries[idx][0] - 1 if idx < len(boundaries) else last_page
        label = f"{num} {title}".strip()
        filename = f"{idx:02d}_{safe_filename(label)}.md"
        chapters.append([filename, label or f"Chapter {idx}", start, end])
    return chapters


def write_chapter_plan(result_dir, output_path):
    first_page, last_page = get_page_bounds(result_dir)
    boundaries = detect_chapters(result_dir)
    if not boundaries:
        print("[Detect] No chapter boundaries found")
        return []
    chapters = build_chapter_plan(boundaries, last_page)
    write_json(output_path, chapters)
    print(f"[Detect] Wrote chapter plan: {output_path}")
    return chapters


def build_page_index(result_dir, output_path, head_lines=5, tail_lines=2, max_line_chars=160):
    """Write a compact page index for review without loading full OCR text into chat."""
    result_dir = Path(result_dir)
    files = sorted(result_dir.glob("*.txt"), key=lambda x: int(x.stem))
    rows = []
    for f in files:
        text = f.read_text(encoding='utf-8').strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        def trim(line):
            return line if len(line) <= max_line_chars else line[:max_line_chars] + '...'

        rows.append({
            'page': int(f.stem),
            'chars': len(text),
            'head': [trim(line) for line in lines[:head_lines]],
            'tail': [trim(line) for line in lines[-tail_lines:]] if tail_lines else [],
        })
    write_json(output_path, rows)
    print(f"[Index] Wrote compact page index: {output_path}")
    return rows


def prepare_llm_cleanup(result_dir, llm_dir, max_chars=12000):
    """Create bounded Markdown chunks for LLM-based cleanup/formatting."""
    result_dir = Path(result_dir)
    llm_dir = Path(llm_dir)
    input_dir = llm_dir / 'input'
    cleaned_dir = llm_dir / 'cleaned'
    input_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(result_dir.glob("*.txt"), key=lambda x: int(x.stem))
    chunks = []
    current = []
    current_pages = []
    current_chars = 0

    def flush():
        nonlocal current, current_pages, current_chars
        if not current:
            return
        chunk_no = len(chunks) + 1
        input_name = f"chunk_{chunk_no:03d}.md"
        output_name = f"chunk_{chunk_no:03d}.md"
        header = [
            "# LLM cleanup chunk",
            "",
            "Clean and format the OCR text as Markdown.",
            "Preserve every `<!-- page: NNN -->` marker exactly.",
            "Do not summarize, omit, or add content.",
            "Fix obvious OCR spacing, paragraph breaks, headings, lists, and table-like text.",
            "",
        ]
        content = '\n'.join(header + current).strip() + '\n'
        (input_dir / input_name).write_text(content, encoding='utf-8')
        chunks.append({
            'chunk': chunk_no,
            'input': str((input_dir / input_name).resolve()),
            'output': str((cleaned_dir / output_name).resolve()),
            'pages': current_pages,
            'chars': current_chars,
        })
        current = []
        current_pages = []
        current_chars = 0

    for f in files:
        page = int(f.stem)
        text = f.read_text(encoding='utf-8').strip()
        block = f"<!-- page: {page:03d} -->\n\n{text}\n"
        if current and current_chars + len(block) > max_chars:
            flush()
        current.append(block)
        current_pages.append(page)
        current_chars += len(block)
    flush()

    manifest = {
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'source_dir': str(result_dir.resolve()),
        'cleaned_dir': str(cleaned_dir.resolve()),
        'max_chars': max_chars,
        'chunks': chunks,
    }
    write_json(llm_dir / 'manifest.json', manifest)
    readme = (
        "# LLM cleanup workspace\n\n"
        "Process files from `input/` one by one with a large model and write the cleaned "
        "Markdown to the matching file in `cleaned/`.\n\n"
        "Rules:\n"
        "- Preserve each `<!-- page: NNN -->` marker exactly.\n"
        "- Do not summarize or delete textbook content.\n"
        "- Improve Markdown headings, paragraphs, lists, formulas, and tables.\n"
        "- Keep chunks independent; do not load the full book into one prompt.\n"
    )
    (llm_dir / 'README.md').write_text(readme, encoding='utf-8')
    print(f"[LLM] Prepared {len(chunks)} cleanup chunks: {llm_dir}")
    return manifest


def apply_llm_cleanup(llm_dir, result_dir):
    """Apply cleaned LLM chunks back to per-page txt files."""
    llm_dir = Path(llm_dir)
    result_dir = Path(result_dir)
    cleaned_dir = llm_dir / 'cleaned'
    if not cleaned_dir.exists():
        raise FileNotFoundError(f"Missing cleaned directory: {cleaned_dir}")

    files = sorted(cleaned_dir.glob("chunk_*.md"))
    if not files:
        raise FileNotFoundError(f"No cleaned chunk_*.md files in {cleaned_dir}")

    result_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    page_re = re.compile(r'<!-- page: (\d{3,}) -->\s*(.*?)(?=<!-- page: \d{3,} -->|\Z)', re.S)
    for f in files:
        text = f.read_text(encoding='utf-8')
        matches = list(page_re.finditer(text))
        if not matches:
            raise ValueError(f"No page markers found in {f}")
        for m in matches:
            page = int(m.group(1))
            body = re.sub(r'<!-- /page: \d{3,} -->', '', m.group(2)).strip()
            (result_dir / f"{page:03d}.txt").write_text(body, encoding='utf-8')
            written += 1
    print(f"[LLM] Applied cleaned chunks to {written} page files")
    return written


# ============================================================
# Main
# ============================================================

def main():
    global OCR_MODEL, OCR_PROMPT, OCR_NUM_CTX, MAX_DIM, IMAGE_ALIGN

    parser = argparse.ArgumentParser(description='Textbook OCR Pipeline')
    parser.add_argument('--image-dir', help='Folder containing numbered image scans')
    parser.add_argument('--output-dir', default='ocr_chapters',
                        help='Output folder for chapter .md files')
    parser.add_argument('--chapters', help='JSON file with chapter definitions')
    parser.add_argument('--work-dir', default='ocr_temp',
                        help='Working directory for intermediate OCR results')
    parser.add_argument('--skip-ocr', action='store_true', help='Skip OCR stage (use existing results)')
    parser.add_argument('--force-ocr', action='store_true', help='Re-OCR all images')
    parser.add_argument('--detect-only', action='store_true',
                        help='Only detect chapter boundaries, then exit')
    parser.add_argument('--auto-chapters', action='store_true',
                        help='Auto-build chapter plan from detected chapter starts')
    parser.add_argument('--chapter-plan-out',
                        help='Write detected chapter plan JSON without loading page text into chat')
    parser.add_argument('--page-index-out',
                        help='Write compact per-page index JSON for boundary review')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help='Parallel Ollama OCR requests. Start with 2; increase only if stable.')
    parser.add_argument('--model', default=OCR_MODEL,
                        help='Ollama model name. Default: TEXTBOOK_OCR_MODEL or glm-ocr')
    parser.add_argument('--max-dim', type=int, default=MAX_DIM,
                        help='Maximum image dimension before OCR downscaling')
    parser.add_argument('--align', type=int, default=IMAGE_ALIGN,
                        help='Resize alignment boundary for OCR images')
    parser.add_argument('--num-ctx', type=int, default=OCR_NUM_CTX,
                        help='Optional Ollama num_ctx value. Use 0 to leave unset')
    parser.add_argument('--prompt', default=OCR_PROMPT,
                        help='OCR prompt sent to the model')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-page OCR progress to stdout')
    parser.add_argument('--progress-every', type=int, default=10,
                        help='In quiet mode, print one OCR progress line every N completed pages')
    parser.add_argument('--log-file',
                        help='Detailed OCR log file. Default: <work-dir>/ocr.log')
    parser.add_argument('--progress-file',
                        help='Machine-readable OCR status JSON. Default: <work-dir>/ocr_progress.json')
    parser.add_argument('--no-auto-ocr-retry', action='store_true',
                        help='Disable the automatic one-pass retry for pages that failed in ocr.log')
    parser.add_argument('--failed-ocr-threshold', type=int, default=DEFAULT_FAIL_PROMPT_THRESHOLD,
                        help='Ask about remaining OCR failures only when count is at or below this number')
    parser.add_argument('--failed-ocr-action',
                        choices=('ask', 'continue', 'retry', 'abort'),
                        default=os.environ.get('TEXTBOOK_OCR_FAILED_ACTION', 'ask'),
                        help='What to do when pages still fail after the automatic retry')
    parser.add_argument('--keep-ollama-on-ocr-skip', action='store_true',
                        help='Do not unload the Ollama OCR model when unresolved failed pages are skipped')
    parser.add_argument('--prepare-llm-cleanup', action='store_true',
                        help='After OCR, write bounded chunks for large-model cleanup and stop')
    parser.add_argument('--apply-llm-cleanup',
                        help='Apply cleaned LLM chunks from this directory back to work-dir pages')
    parser.add_argument('--llm-dir',
                        help='LLM cleanup workspace. Default: <work-dir>/llm_cleanup')
    parser.add_argument('--llm-chunk-chars', type=int, default=12000,
                        help='Approximate max characters per LLM cleanup chunk')
    parser.add_argument('--skip-local-clean', action='store_true',
                        help='Skip local regex cleanup')
    parser.add_argument('--skip-local-format', action='store_true',
                        help='Skip local Markdown formatting')
    parser.add_argument('--local-after-llm', action='store_true',
                        help='Run local clean/format even after applying LLM-cleaned chunks')

    args = parser.parse_args()
    if args.failed_ocr_action not in {'ask', 'continue', 'retry', 'abort'}:
        parser.error('--failed-ocr-action must be one of: ask, continue, retry, abort')

    OCR_MODEL = args.model
    OCR_PROMPT = args.prompt
    OCR_NUM_CTX = max(0, int(args.num_ctx or 0))
    MAX_DIM = max(64, int(args.max_dim or 1600))
    IMAGE_ALIGN = max(1, int(args.align or 16))

    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    log_file = Path(args.log_file) if args.log_file else work_dir / 'ocr.log'
    progress_file = Path(args.progress_file) if args.progress_file else work_dir / 'ocr_progress.json'
    llm_dir = Path(args.llm_dir) if args.llm_dir else work_dir / 'llm_cleanup'

    if args.detect_only:
        if args.page_index_out:
            build_page_index(work_dir, args.page_index_out)
        if args.chapter_plan_out:
            write_chapter_plan(work_dir, args.chapter_plan_out)
        else:
            detect_chapters(work_dir)
        return

    # Stage 1: OCR
    if not args.skip_ocr:
        if not args.image_dir:
            parser.error('--image-dir is required unless --skip-ocr or --detect-only is used')
        log_start_offset = log_file.stat().st_size if log_file.exists() else 0
        batch_ocr(
            args.image_dir,
            work_dir,
            force=args.force_ocr,
            workers=args.workers,
            quiet=not args.verbose,
            progress_every=args.progress_every,
            log_file=log_file,
            progress_file=progress_file,
        )
        failures = read_ocr_failures_from_log(log_file, log_start_offset)
        if failures and not args.no_auto_ocr_retry:
            retry_start_offset = log_file.stat().st_size if log_file.exists() else 0
            retry_failed_pages_once(
                args.image_dir,
                work_dir,
                failures,
                quiet=not args.verbose,
                progress_every=args.progress_every,
                log_file=log_file,
                progress_file=progress_file,
            )
            failures = read_ocr_failures_from_log(log_file, retry_start_offset)

        if failures:
            failed_threshold = max(0, int(args.failed_ocr_threshold))
            print_failed_page_report(failures, max_pages=failed_threshold)
            if should_retry_after_report(
                failures,
                failed_threshold,
                args.failed_ocr_action,
            ):
                retry_start_offset = log_file.stat().st_size if log_file.exists() else 0
                retry_failed_pages_once(
                    args.image_dir,
                    work_dir,
                    failures,
                    quiet=not args.verbose,
                    progress_every=args.progress_every,
                    log_file=log_file,
                    progress_file=progress_file,
                )
                failures = read_ocr_failures_from_log(log_file, retry_start_offset)
                if failures:
                    print_failed_page_report(failures, max_pages=failed_threshold)
                    if not args.keep_ollama_on_ocr_skip:
                        stop_ollama_model(log_file)
            elif not args.keep_ollama_on_ocr_skip:
                stop_ollama_model(log_file)

    if args.prepare_llm_cleanup:
        prepare_llm_cleanup(work_dir, llm_dir, max_chars=args.llm_chunk_chars)
        print("[LLM] Cleanup chunks are ready. Clean input/*.md into cleaned/*.md, then rerun with --apply-llm-cleanup.")
        return

    applied_llm = False
    if args.apply_llm_cleanup:
        apply_llm_cleanup(args.apply_llm_cleanup, work_dir)
        applied_llm = True

    # Stage 2: Clean
    run_local_clean = not args.skip_local_clean and (not applied_llm or args.local_after_llm)
    if run_local_clean:
        clean_all(work_dir)
    else:
        print("[Clean] Skipped local cleanup")

    # Stage 3: Markdown formatting
    run_local_format = not args.skip_local_format and (not applied_llm or args.local_after_llm)
    if run_local_format:
        format_all(work_dir)
    else:
        print("[Format] Skipped local formatting")

    if args.page_index_out:
        build_page_index(work_dir, args.page_index_out)

    # Stage 4: Merge
    if args.chapters:
        with open(args.chapters, 'r', encoding='utf-8') as f:
            chapters = json.load(f)
    elif args.auto_chapters or args.chapter_plan_out:
        plan_path = args.chapter_plan_out or (work_dir / 'chapters.detected.json')
        chapters = write_chapter_plan(work_dir, plan_path)
        if not args.auto_chapters:
            print("[Merge] Chapter plan written. Review it, then rerun with --chapters.")
            return
    else:
        print("[Merge] No chapters file specified. Use --auto-chapters or --chapter-plan-out.")
        return

    if not chapters:
        print("[Merge] No chapters to merge.")
        return

    merge_chapters(work_dir, output_dir, chapters)
    print("\nPipeline complete!")


if __name__ == '__main__':
    main()
