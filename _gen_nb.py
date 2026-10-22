#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build PersianEmoTTS.ipynb — a self-contained Colab notebook."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
ENGINE = (ROOT / "src" / "persian_emo_tts.py").read_text(encoding="utf-8")

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
}

cells = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src.strip() + "\n"))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip() + "\n"))


# =============================================================================
md(r"""
# PersianEmoTTS
## تبدیل متن فارسی به گفتار احساسی
### Emotion-Controllable Persian Text-to-Speech

نوت‌بوک نهایی پروژه: کاربر یک جملهٔ فارسی می‌نویسد و سامانه گفتاری **واضح، فارسی و احساسی** می‌سازد.

این نوت‌بوک روی **Google Colab (GPU T4 توصیه‌شده، CPU هم کار می‌کند)** از بالا به پایین اجرا می‌شود.  
`Runtime → Run all`

---

### ایدهٔ پژوهشی (هم‌راستا با مقالات ریپو)

| مقاله | نقش در این پروژه |
|---|---|
| **TTS-CtrlNet** (Jeong et al. 2025) | ستون فقرات TTS منجمد می‌ماند؛ احساس با یک **Control Scale λ** روی شاخهٔ کنترل اعمال می‌شود |
| **StyleTTS 2** (Li et al. 2023) | احساس به‌صورت **بردار استایل** (F0، انرژی، دیرش، کیفیت صدا) مدل می‌شود |
| **EmoVoice** (Yang et al. 2025) | برچسب/پرامپت احساس + ارزیابی با SER و شباهت احساسی |
| **AutESD** (Rahmati et al. 2026) | چهار احساس اصلی فارسی: خنثی / شادی / خشم / اندوه + پروتکل UAR |
| **PersianVox** (Zouashkiani et al. 2026) | اهمیت پروزودی و کیفیت گفتار فارسی |
| نوت‌بوک‌های استاد | **Piper Mana/Ganji** + G2P + بنچ‌مارک RTF + رابط Gradio |

### معماری

```
متن فارسی
   │
   ├─ Frontend: نرمال‌سازی، ارقام، تشخیص احساس از متن
   ├─ (اختیاری) Homo-GE2PE  →  واج‌آرایی دقیق
   │
   ▼
Piper منجمد  (Mana زنانه / Ganji مردانه)     ← کیفیت بالای فارسی
   │
   ▼
Emotion Controller  λ ∈ [0, 1.25]
   • length / noise Piper
   • PSOLA (Praat): F0، دامنه، فرمنت، کانتور
   • انرژی، درخشندگی، jitter، مکث
   │
   ▼
WAV احساسی  +  ارزیابی (آکوستیک / SER / RTF / CER)
```

چهار احساس AutESD همیشه ساخته می‌شوند؛ ترس و شگفتی (ShEMO) هم فعال‌اند.
""")

# =============================================================================
code(r"""
# ============================================================
# Cell 0: Environment Check
# PersianEmoTTS
# ============================================================

import os, sys, platform, shutil, subprocess
from pathlib import Path

print("=" * 70)
print("PersianEmoTTS — Environment Check")
print("=" * 70)

print("\n[1] Python / OS")
print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
print("CWD:", os.getcwd())

print("\n[2] Colab")
try:
    import google.colab  # type: ignore
    IN_COLAB = True
except Exception:
    IN_COLAB = False
print("Running in Google Colab:", IN_COLAB)

print("\n[3] PyTorch / CUDA / GPU")
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {p.name}  {p.total_memory/1024**3:.2f} GB")
except Exception as e:
    print("PyTorch not ready yet:", repr(e))

print("\n[4] nvidia-smi")
try:
    r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(r.stdout if r.returncode == 0 else r.stderr)
except Exception:
    print("nvidia-smi not found (CPU is fine for Piper).")

print("\n[5] RAM / Disk")
try:
    import psutil
    ram = psutil.virtual_memory()
    print(f"RAM total {ram.total/1024**3:.2f} GB | available {ram.available/1024**3:.2f} GB")
except Exception:
    pass
for pth in [os.getcwd(), "/content", "/"]:
    if os.path.isdir(pth):
        t, u, f = shutil.disk_usage(pth)
        print(f"{pth}: free {f/1024**3:.2f} GB / {t/1024**3:.2f} GB")
        break

print("\n" + "=" * 70)
print("Environment check finished.")
print("=" * 70)
""")

# =============================================================================
code(r"""
# ============================================================
# Cell 1: Install dependencies
# ============================================================

import os, sys, subprocess, shutil
from pathlib import Path

def pip_install(*pkgs):
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + list(pkgs)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:])
        print(r.stderr[-2500:])
        raise RuntimeError("pip failed: " + " ".join(pkgs))

print("[1] System packages (espeak-ng, ffmpeg, libsndfile)")
if shutil.which("apt-get"):
    subprocess.run(["apt-get", "update", "-qq"], check=False)
    subprocess.run(
        ["apt-get", "install", "-y", "-qq", "espeak-ng", "ffmpeg", "libsndfile1"],
        check=False,
    )

print("[2] Python packages")
pip_install(
    "piper-tts", "soundfile", "librosa", "praat-parselmouth", "hazm",
    "gradio", "scikit-learn", "pandas", "matplotlib", "scipy",
    "datasets", "transformers", "torchaudio", "unidecode", "pydub",
    "accelerate",
)

# optional alignment extra used in the professor notebook
try:
    pip_install("piper-tts[alignment]")
except Exception as e:
    print("piper alignment extra skipped:", e)

print("[3] Import smoke-test")
import numpy, soundfile, librosa, sklearn, pandas, matplotlib, scipy
print("numpy", numpy.__version__, "| sklearn", sklearn.__version__)
try:
    get_ipython().run_line_magic("matplotlib", "inline")  # type: ignore
except Exception:
    pass
try:
    import parselmouth
    print("parselmouth", parselmouth.__version__)
except Exception as e:
    print("WARN parselmouth:", e)
try:
    from piper import PiperVoice
    print("piper: OK")
except Exception as e:
    print("WARN piper:", e)

print("Install finished.")
""")

# =============================================================================
code(r"""
# ============================================================
# Cell 2: Project directories + config
# ============================================================

import json, os, sys
from pathlib import Path
from datetime import datetime

try:
    import google.colab  # type: ignore
    IN_COLAB = True
except Exception:
    IN_COLAB = False

ROOT = Path("/content") if IN_COLAB else Path.cwd()
PROJECT_DIR = ROOT / "PersianEmoTTS_Project"
for sub in ["src", "models/piper", "samples/demo", "samples/benchmark",
            "samples/ui", "reports", "configs", "data/shemo"]:
    (PROJECT_DIR / sub).mkdir(parents=True, exist_ok=True)

if str(PROJECT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "src"))
if str(Path.cwd() / "src") not in sys.path:
    sys.path.insert(0, str(Path.cwd() / "src"))

project_config = {
    "created_at": datetime.now().isoformat(),
    "name": "PersianEmoTTS",
    "architecture": "Frozen Persian Piper + Emotion ControlNet-style style vector + Praat PSOLA",
    "emotions_autesd": ["neutral", "happy", "angry", "sad"],
    "emotions_shemo_extra": ["fear", "surprise"],
    "voices": ["female/Mana", "male/Ganji"],
    "in_colab": IN_COLAB,
    "project_dir": str(PROJECT_DIR),
}
cfg_path = PROJECT_DIR / "project_config.json"
cfg_path.write_text(json.dumps(project_config, ensure_ascii=False, indent=2), encoding="utf-8")

print("PROJECT_DIR:", PROJECT_DIR)
print("Config saved:", cfg_path)
print(json.dumps(project_config, ensure_ascii=False, indent=2))
""")

# =============================================================================
import base64 as _b64
_ENGINE_B64 = _b64.b64encode(ENGINE.encode("utf-8")).decode("ascii")

code(
    "# ============================================================\n"
    "# Cell 3: Write PersianEmoTTS engine module\n"
    "# ============================================================\n\n"
    "from pathlib import Path\n"
    "import sys, base64\n\n"
    "PROJECT_DIR = Path(\"/content/PersianEmoTTS_Project\") if Path(\"/content/PersianEmoTTS_Project\").exists() else Path.cwd() / \"PersianEmoTTS_Project\"\n"
    "SRC = PROJECT_DIR / \"src\"\n"
    "SRC.mkdir(parents=True, exist_ok=True)\n"
    "target = SRC / \"persian_emo_tts.py\"\n\n"
    "candidates = [\n"
    "    Path.cwd() / \"src\" / \"persian_emo_tts.py\",\n"
    "    Path(\"/content/tts/src/persian_emo_tts.py\"),\n"
    "    Path(\"/content/PersianEmoTTS_Project/src/persian_emo_tts.py\"),\n"
    "]\n"
    "copied = False\n"
    "for c in candidates:\n"
    "    if c.exists() and c.resolve() != target.resolve() and c.stat().st_size > 1000:\n"
    "        target.write_text(c.read_text(encoding=\"utf-8\"), encoding=\"utf-8\")\n"
    "        print(\"Engine copied from\", c)\n"
    "        copied = True\n"
    "        break\n\n"
    "if not copied:\n"
    f"    ENGINE_B64 = \"{_ENGINE_B64}\"\n"
    "    target.write_bytes(base64.b64decode(ENGINE_B64))\n"
    "    print(\"Engine written from embedded source.\")\n\n"
    "if str(SRC) not in sys.path:\n"
    "    sys.path.insert(0, str(SRC))\n\n"
    "import importlib\n"
    "import persian_emo_tts as emotts\n"
    "importlib.reload(emotts)\n"
    "print(\"Engine OK — emotions:\", list(emotts.STYLE_PRIORS))\n"
    "print(\"Module:\", target, \"size\", target.stat().st_size, \"bytes\")\n"
)

# =============================================================================
code(r"""
# ============================================================
# Cell 4: Download Piper voices (Mana female + Ganji male)
# ============================================================

import os, sys, urllib.request
from pathlib import Path

PROJECT_DIR = Path("/content/PersianEmoTTS_Project") if Path("/content/PersianEmoTTS_Project").exists() else Path.cwd() / "PersianEmoTTS_Project"
MODEL_DIR = PROJECT_DIR / "models" / "piper"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "fa_IR-mana-medium.onnx":
        "https://huggingface.co/MahtaFetrat/Mana-Persian-Piper/resolve/main/fa_IR-mana-medium.onnx",
    "fa_IR-mana-medium.onnx.json":
        "https://huggingface.co/MahtaFetrat/Mana-Persian-Piper/resolve/main/fa_IR-mana-medium.onnx.json",
    "fa_IR-ganji-medium.onnx":
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR/ganji/medium/fa_IR-ganji-medium.onnx",
    "fa_IR-ganji-medium.onnx.json":
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR/ganji/medium/fa_IR-ganji-medium.onnx.json",
}

def download(url, dest: Path):
    if dest.exists() and dest.stat().st_size > 50_000:
        print(f"exists {dest.name} ({dest.stat().st_size/1024**2:.1f} MB)")
        return
    print("downloading", dest.name, "...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print("urllib failed, trying wget:", e)
        import subprocess
        subprocess.run(["wget", "-q", "-O", str(dest), url], check=True)
    print(f"  -> {dest.stat().st_size/1024**2:.1f} MB")

for name, url in FILES.items():
    download(url, MODEL_DIR / name)

FEMALE_ONNX = MODEL_DIR / "fa_IR-mana-medium.onnx"
FEMALE_JSON = MODEL_DIR / "fa_IR-mana-medium.onnx.json"
MALE_ONNX   = MODEL_DIR / "fa_IR-ganji-medium.onnx"
MALE_JSON   = MODEL_DIR / "fa_IR-ganji-medium.onnx.json"
assert FEMALE_ONNX.exists() and MALE_ONNX.exists()
print("Mana (female):", FEMALE_ONNX)
print("Ganji (male) :", MALE_ONNX)
""")

# =============================================================================
code(r"""
# ============================================================
# Cell 5: Load frozen Piper backbone
# ============================================================

import sys
from pathlib import Path
import importlib

PROJECT_DIR = Path("/content/PersianEmoTTS_Project") if Path("/content/PersianEmoTTS_Project").exists() else Path.cwd() / "PersianEmoTTS_Project"
sys.path.insert(0, str(PROJECT_DIR / "src"))
import persian_emo_tts as emotts
importlib.reload(emotts)

MODEL_DIR = PROJECT_DIR / "models" / "piper"
FEMALE_ONNX = MODEL_DIR / "fa_IR-mana-medium.onnx"
FEMALE_JSON = MODEL_DIR / "fa_IR-mana-medium.onnx.json"
MALE_ONNX   = MODEL_DIR / "fa_IR-ganji-medium.onnx"
MALE_JSON   = MODEL_DIR / "fa_IR-ganji-medium.onnx.json"

tts = emotts.PersianEmoTTS()
print("Loading Mana (female) ...")
tts.backbone.load("female", FEMALE_ONNX, FEMALE_JSON)
print("Loading Ganji (male) ...")
tts.backbone.load("male", MALE_ONNX, MALE_JSON)
print("Sample rate:", tts.backbone.sample_rate)
print("Voices loaded:", list(tts.backbone.voices))

# quick sanity synth
res = tts.synthesize("سلام. این یک آزمایش کوتاه است.", emotion="neutral", intensity=0.0, gender="female")
print(f"Sanity: {res.duration:.2f}s  RTF={res.rtf:.3f}  sr={res.sr}")
from IPython.display import Audio, display
display(Audio(res.audio, rate=res.sr))
""")

# =============================================================================
md(r"""
## کنترل احساس

ستون فقرات Piper **یخ‌زده** است (کیفیت فارسی حفظ می‌شود).  
بردار استایل هر احساس با **شدت λ** بین خنثی و احساس هدف درون‌یابی می‌شود، بعد با **Praat PSOLA** روی F0/فرمنت/کانتور و سپس انرژی و درخشندگی اعمال می‌گردد.

جملهٔ آزمون باید از نظر واژگانی تا حد ممکن خنثی باشد تا احساس از **صدا** بیاید نه از معنی جمله — مطابق پروتکل AutESD.
""")

# =============================================================================
code(r"""
# ============================================================
# Cell 6: Same-text emotion demo  (listen before anything else)
# ============================================================
# جمله از نظر محتوا خنثی است تا احساس فقط در صدا شنیده شود.

import json
from pathlib import Path
from IPython.display import Audio, display, Markdown
import soundfile as sf
import numpy as np

PROJECT_DIR = Path("/content/PersianEmoTTS_Project") if Path("/content/PersianEmoTTS_Project").exists() else Path.cwd() / "PersianEmoTTS_Project"
DEMO_DIR = PROJECT_DIR / "samples" / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

DEMO_TEXT = "فردا ساعت سه در میدان اصلی شهر منتظر شما می‌مانم."
INTENSITY = 0.95
GENDER = "female"   # "female" or "male"
EMOS = ["neutral", "happy", "angry", "sad", "fear", "surprise"]

print("متن:", DEMO_TEXT)
print("شدت λ =", INTENSITY, "| گوینده =", GENDER)
print("=" * 70)

demo_rows = []
for emo in EMOS:
    r = tts.synthesize(DEMO_TEXT, emotion=emo, intensity=INTENSITY, gender=GENDER)
    wav_path = DEMO_DIR / f"demo_{GENDER}_{emo}.wav"
    emotts.save_wav(wav_path, r.audio, r.sr)
    feat = emotts.extract_acoustic_features(r.audio, r.sr)
    demo_rows.append((emo, r, feat, wav_path))
    fa = emotts.EMOTION_FA[emo]
    display(Markdown(
        f"### {fa} (`{emo}`) — {r.duration:.2f}s — RTF {r.rtf:.3f} — "
        f"F0̄={feat.get('f0_mean',0):.1f} Hz — RMS={feat.get('rms',0):.3f}"
    ))
    display(Audio(r.audio, rate=r.sr))

print("\nفایل‌ها در", DEMO_DIR)
""")

# =============================================================================
code(r"""
# ============================================================
# Cell 7: Acoustic gates — must PASS for emotion to be real
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
from IPython.display import display

by = {}
rows = []
for emo, r, feat, wav_path in demo_rows:
    by.setdefault(emo, []).append(feat)
    rows.append({
        "emotion": emo,
        "duration_s": feat["duration"],
        "f0_mean_Hz": feat["f0_mean"],
        "f0_std_Hz": feat["f0_std"],
        "f0_range_Hz": feat["f0_range"],
        "rms": feat["rms"],
        "centroid_Hz": feat.get("centroid", 0.0),
        "rtf": r.rtf,
    })

df_demo = pd.DataFrame(rows).set_index("emotion")
display(df_demo.round(3))

gates = emotts.acoustic_gates(by)
print("\n" + "=" * 70)
print("ACOUSTIC EMOTION GATES")
print("=" * 70)
for k, v in gates.items():
    if k == "means":
        continue
    print(f"  {k:24s} {v}")
print("means:", {k: (None if v is None else round(v, 3)) for k, v in gates["means"].items()})

if not gates["passed"]:
    raise RuntimeError(
        "Acoustic emotion gates FAILED — احساس در سیگنال جدا نشده. "
        "شدت λ را بالا ببرید یا parselmouth را بررسی کنید."
    )

print("\n✓  احساس در سطح آکوستیک روی جملهٔ فارسی جدا شده است.")
print("    (F0 شادی/خشم > اندوه، دیرش اندوه > خشم، انرژی خشم > اندوه)")
""")

# =============================================================================
md(r"""
## دیتاست احساسی فارسی و SER

AutESD هنوز به‌صورت عمومی منتشر نشده؛ خود مقاله ShEMO را به‌عنوان منبع قبلی فارسی معرفی می‌کند.  
اینجا از **ShEMO** (۳۰۰۰ گفته، ۸۷ گوینده، ۶ احساس) برای:

1. آمار استایل واقعی فارسی (F0 / انرژی / دیرش)
2. آموزش SVM روی MFCC+F0 — همان خانوادهٔ ارزیابی AutESD (KNN/SVM/CNN)

اگر دانلود ShEMO شکست بخورد، ارزیابی آکوستیک سلول ۷ به‌تنهایی کافی است و نوت‌بوک متوقف نمی‌شود.
""")

# =============================================================================
code(r"""
# ============================================================
# Cell 8: ShEMO subset + SVM speech-emotion recogniser
# AutESD-style features (MFCC + F0 + energy) + linear SVM
# ============================================================

import os, re, json, traceback
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_DIR = Path("/content/PersianEmoTTS_Project") if Path("/content/PersianEmoTTS_Project").exists() else Path.cwd() / "PersianEmoTTS_Project"
SHEMO_DIR = PROJECT_DIR / "data" / "shemo"
SHEMO_DIR.mkdir(parents=True, exist_ok=True)

ser_bundle = {"available": False, "clf": None, "labels": list(emotts.AUTESD_EMOTIONS)}
shemo_stats = {}

def parse_shemo_label(name: str):
    # F03S02 / M12A01  -> emotion letter is the first A/H/N/S/W after speaker id
    m = re.search(r"[FM]\d{2}([AHNSWF])", name.upper())
    if not m:
        return None
    return emotts.SHEMO_CODE.get(m.group(1))

print("[1] Trying HuggingFace minoosh/shEMO ...")
X, y = [], []
try:
    from datasets import load_dataset, Audio
    # 30% is enough for a compact SVM and keeps Colab RAM happy
    ds = load_dataset("minoosh/shEMO", split="train[:35%]")
    # column names vary; be defensive
    cols = ds.column_names
    print("columns:", cols)
    n_ok = 0
    cap = {e: 0 for e in emotts.EMOTION_ORDER}
    CAP_PER = 70
    for row in ds:
        emo = None
        if "emotion" in row:
            raw = row["emotion"]
            if isinstance(raw, int):
                # minoosh mapping: A0 H1 N2 S3 W4 F5  (from HF card)
                inv = {0: "angry", 1: "happy", 2: "neutral", 3: "sad", 4: "surprise", 5: "fear"}
                emo = inv.get(raw)
            else:
                emo = emotts.SHEMO_CODE.get(str(raw).upper()[:1], str(raw).lower())
        if emo not in cap or cap[emo] >= CAP_PER:
            continue
        audio = row.get("audio") or row.get("speech")
        if audio is None:
            continue
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        if arr.size < sr * 0.4:
            continue
        feat = emotts.extract_acoustic_features(arr, sr)
        X.append(emotts.features_to_vector(feat))
        y.append(emo)
        cap[emo] += 1
        n_ok += 1
        shemo_stats.setdefault(emo, []).append(feat)
    print("loaded", n_ok, "clips  per-class", dict(cap))
except Exception as e:
    print("ShEMO download/parse failed (continuing without SVM-SER).")
    traceback.print_exc()

if len(X) >= 40 and len(set(y)) >= 3:
    X = np.stack(X)
    y = np.array(y)
    labels = sorted(set(y))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    try:
        svm = LinearSVC(class_weight="balanced", max_iter=4000, dual="auto")
    except TypeError:
        svm = LinearSVC(class_weight="balanced", max_iter=4000)
    clf = make_pipeline(StandardScaler(), svm)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    print("\n[2] SVM-SER on held-out ShEMO")
    print(classification_report(yte, pred, digits=3))
    print("confusion\n", confusion_matrix(yte, pred, labels=labels))
    ser_bundle.update(available=True, clf=clf, labels=labels)
    print("SVM-SER is ready and will be used on generated TTS.")
else:
    print("[2] Not enough ShEMO rows — SVM-SER disabled. Acoustic gates still apply.")

# summarise ShEMO acoustics if we have them
if shemo_stats:
    print("\n[3] ShEMO acoustic means (subset)")
    for emo, feats in shemo_stats.items():
        f0 = np.mean([f["f0_mean"] for f in feats if f["f0_mean"] > 0] or [0])
        du = np.mean([f["duration"] for f in feats])
        rms = np.mean([f["rms"] for f in feats])
        print(f"  {emo:10s}  F0̄={f0:6.1f} Hz  dur={du:.2f}s  rms={rms:.3f}  n={len(feats)}")
""")

# =============================================================================
code(r"""
# ============================================================
# Cell 9: Full benchmark
# Neutral-text × AutESD emotions × 2 speakers
# Metrics: RTF, acoustic gates, SVM-SER UAR, Whisper CER
# ============================================================

import time, json, traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Audio, Markdown

PROJECT_DIR = Path("/content/PersianEmoTTS_Project") if Path("/content/PersianEmoTTS_Project").exists() else Path.cwd() / "PersianEmoTTS_Project"
BENCH_DIR = PROJECT_DIR / "samples" / "benchmark"
BENCH_DIR.mkdir(parents=True, exist_ok=True)
REPORTS = PROJECT_DIR / "reports"

SENTENCES = emotts.NEUTRAL_EVAL_SENTENCES[:5]
EMOS = list(emotts.AUTESD_EMOTIONS)          # AutESD protocol
GENDERS = ["female", "male"]
INTENSITY = 0.95

records = []
by_emo_feat = defaultdict(list)
print(f"{len(SENTENCES)} sentences × {len(EMOS)} emotions × {len(GENDERS)} speakers")
print("=" * 70)

t_all = time.time()
for gi, gender in enumerate(GENDERS):
    for si, sent in enumerate(SENTENCES):
        for emo in EMOS:
            r = tts.synthesize(sent, emotion=emo, intensity=INTENSITY, gender=gender)
            feat = emotts.extract_acoustic_features(r.audio, r.sr)
            by_emo_feat[emo].append(feat)
            wav = BENCH_DIR / f"{gender}_s{si}_{emo}.wav"
            emotts.save_wav(wav, r.audio, r.sr)
            ser_pred, ser_ok = None, None
            if ser_bundle.get("available"):
                vec = emotts.features_to_vector(feat).reshape(1, -1)
                ser_pred = str(ser_bundle["clf"].predict(vec)[0])
                ser_ok = (ser_pred == emo)
            records.append({
                "gender": gender, "sent_id": si, "text": sent, "emotion": emo,
                "duration": r.duration, "rtf": r.rtf, "synth_s": r.synth_seconds,
                "f0_mean": feat["f0_mean"], "f0_std": feat["f0_std"],
                "rms": feat["rms"], "centroid": feat.get("centroid", 0.0),
                "ser_pred": ser_pred, "ser_ok": ser_ok, "wav": str(wav),
            })
        print(f"  {gender} sentence {si+1}/{len(SENTENCES)} done")

bench_df = pd.DataFrame(records)
display(bench_df.groupby(["emotion", "gender"])[["duration", "rtf", "f0_mean", "rms"]].mean().round(3))

print("\n[1] RTF")
print(f"  mean RTF   : {bench_df.rtf.mean():.4f}")
print(f"  median RTF : {bench_df.rtf.median():.4f}")
print(f"  realtime × : {1.0/max(bench_df.rtf.mean(), 1e-8):.1f}")

print("\n[2] Acoustic gates on the full set")
gates2 = emotts.acoustic_gates(by_emo_feat)
for k, v in gates2.items():
    if k != "means":
        print(f"  {k:24s} {v}")
assert gates2["passed"], "Full-set acoustic gates failed."

print("\n[3] SVM-SER (trained on ShEMO, tested on our TTS)")
if ser_bundle.get("available") and bench_df.ser_pred.notna().any():
    sub = bench_df.dropna(subset=["ser_pred"])
    uar = emotts.uar(list(sub.emotion), list(sub.ser_pred), list(emotts.AUTESD_EMOTIONS))
    acc = float(sub.ser_ok.mean())
    print(f"  Accuracy={acc:.3f}  UAR={uar:.3f}  (chance ≈ 0.25)")
    print(pd.crosstab(sub.emotion, sub.ser_pred, rownames=["true"], colnames=["pred"]))
    # Cross-domain UAR is expected to be well above chance but not perfect:
    # ShEMO is acted radio-play, our TTS is read speech + PSOLA.
    ser_ok_flag = uar >= 0.35
else:
    uar, acc, ser_ok_flag = None, None, None
    print("  SVM-SER not available — skipped.")

print("\n[4] Whisper CER (intelligibility) on a small slice")
cer_values = []
try:
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=device,
    )
    # one sentence × 4 emotions, female only, to keep runtime reasonable
    slice_df = bench_df[(bench_df.gender == "female") & (bench_df.sent_id == 0)]
    def _norm(s):
        return emotts.normalize_persian(s).replace(" ", "")
    for _, row in slice_df.iterrows():
        try:
            hyp = asr(row["wav"], generate_kwargs={"language": "fa", "task": "transcribe"})["text"]
        except TypeError:
            hyp = asr(row["wav"])["text"]
        ref, hypn = _norm(row["text"]), _norm(hyp)
        # char error rate
        import difflib
        sm = difflib.SequenceMatcher(a=ref, b=hypn)
        cer = 1.0 - sm.ratio()
        cer_values.append(cer)
        print(f"  {row.emotion:8s} CER={cer:.2f}  hyp={hyp[:80]}")
    print(f"  mean CER = {np.mean(cer_values):.3f}  (lower is better; <0.45 is acceptable for Whisper-small FA)")
except Exception as e:
    print("  Whisper skipped:", repr(e))

# plots
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
order = list(EMOS)
for ax, col, title in zip(
    axes,
    ["f0_mean", "rms", "duration"],
    ["F0 mean (Hz)", "Energy (RMS)", "Duration (s)"],
):
    data = [bench_df[bench_df.emotion == e][col].values for e in order]
    ax.boxplot(data)
    ax.set_xticklabels([emotts.EMOTION_FA[e] for e in order])
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
fig.suptitle("PersianEmoTTS — same-text acoustic separation (AutESD 4-class)", y=1.03)
fig.tight_layout()
fig_path = REPORTS / "acoustic_separation.png"
fig.savefig(fig_path, dpi=140, bbox_inches="tight")
plt.show()
print("figure:", fig_path)

bench_path = REPORTS / "benchmark_rows.csv"
bench_df.to_csv(bench_path, index=False)
print("csv:", bench_path)
print(f"\nBenchmark wall time: {time.time()-t_all:.1f}s")
""")

# =============================================================================
code(r"""
# ============================================================
# Cell 10: Congruent-emotion samples (feeling + wording agree)
# ============================================================

from IPython.display import Audio, display, Markdown
from pathlib import Path

PROJECT_DIR = Path("/content/PersianEmoTTS_Project") if Path("/content/PersianEmoTTS_Project").exists() else Path.cwd() / "PersianEmoTTS_Project"
OUT = PROJECT_DIR / "samples" / "demo"
OUT.mkdir(parents=True, exist_ok=True)

print("نمونه‌های هم‌خوان (متن و احساس یکی هستند) — گوینده زنانه، λ=1.0")
for emo, sent in emotts.CONGRUENT_SENTENCES.items():
    r = tts.synthesize(sent, emotion=emo, intensity=1.0, gender="female")
    emotts.save_wav(OUT / f"congruent_{emo}.wav", r.audio, r.sr)
    display(Markdown(f"**{emotts.EMOTION_FA[emo]}** — {sent}"))
    display(Audio(r.audio, rate=r.sr))
""")

# =============================================================================
md(r"""
## رابط کاربری

متن را بنویسید، احساس را انتخاب کنید (یا روی خودکار بگذارید تا از روی خود جمله تشخیص داده شود)، شدت را تنظیم کنید و گفتار را بشنوید.
""")

# =============================================================================
code(r"""
# ============================================================
# Cell 11: Gradio UI
# ============================================================

import uuid, time, glob, os
from pathlib import Path
import numpy as np
import gradio as gr

PROJECT_DIR = Path("/content/PersianEmoTTS_Project") if Path("/content/PersianEmoTTS_Project").exists() else Path.cwd() / "PersianEmoTTS_Project"
UI_DIR = PROJECT_DIR / "samples" / "ui"
UI_DIR.mkdir(parents=True, exist_ok=True)

EMO_LABELS = {
    "خودکار از روی متن": "auto",
    "خنثی": "neutral",
    "شادی": "happy",
    "خشم": "angry",
    "اندوه": "sad",
    "ترس": "fear",
    "شگفتی": "surprise",
}
GENDER_LABELS = {"زنانه — Mana": "female", "مردانه — Ganji": "male"}

def _cleanup():
    now = time.time()
    for f in UI_DIR.glob("ui_*.wav"):
        try:
            if now - f.stat().st_mtime > 600:
                f.unlink()
        except OSError:
            pass

def generate_ui(text, emo_label, gender_label, intensity):
    if not text or not str(text).strip():
        return None, "متن خالی است."
    _cleanup()
    emo_key = EMO_LABELS.get(emo_label, emo_label)
    gender = GENDER_LABELS.get(gender_label, gender_label)
    r = tts.synthesize(str(text), emotion=emo_key, intensity=float(intensity), gender=gender)
    wav = UI_DIR / f"ui_{r.emotion}_{uuid.uuid4().hex[:8]}.wav"
    emotts.save_wav(wav, r.audio, r.sr)
    feat = emotts.extract_acoustic_features(r.audio, r.sr)
    ser_line = ""
    if ser_bundle.get("available"):
        pred = ser_bundle["clf"].predict(emotts.features_to_vector(feat).reshape(1, -1))[0]
        ser_line = f"پیش‌بینی SER (SVM/ShEMO): {emotts.EMOTION_FA.get(pred, pred)}\n"
    report = (
        f"متن نرمال‌شده : {r.normalized}\n"
        f"احساس اعمال‌شده : {emotts.EMOTION_FA.get(r.emotion, r.emotion)}  ({r.emotion})\n"
        f"تشخیص از متن     : {emotts.EMOTION_FA.get(r.text_emotion, r.text_emotion)}  "
        f"( conf={r.text_emotion_conf:.2f} )\n"
        f"شدت λ             : {r.intensity:.2f}\n"
        f"گوینده            : {r.gender}\n"
        f"مدت               : {r.duration:.2f} s\n"
        f"RTF               : {r.rtf:.3f}  ({1.0/max(r.rtf,1e-8):.1f}× realtime)\n"
        f"F0 میانگین        : {feat.get('f0_mean',0):.1f} Hz\n"
        f"انرژی RMS         : {feat.get('rms',0):.3f}\n"
        f"{ser_line}"
        f"واج‌ها             : {r.phonemes or '— (متن خام Piper)'}\n"
    )
    return str(wav), report

with gr.Blocks(title="PersianEmoTTS") as demo:
    gr.Markdown(
        "# PersianEmoTTS\n"
        "تبدیل متن فارسی به گفتار احساسی — Piper منجمد + کنترل استایل (TTS-CtrlNet / StyleTTS2)"
    )
    with gr.Row():
        inp = gr.Textbox(
            label="متن فارسی",
            lines=3,
            value="فردا ساعت سه در میدان اصلی شهر منتظر شما می‌مانم.",
        )
    with gr.Row():
        emo = gr.Radio(
            choices=list(EMO_LABELS.keys()), value="خودکار از روی متن", label="احساس",
        )
        gender = gr.Radio(choices=list(GENDER_LABELS.keys()), value="زنانه — Mana", label="گوینده")
        intensity = gr.Slider(0.0, 1.25, value=0.95, step=0.05, label="شدت احساس λ")
    btn = gr.Button("تولید گفتار", variant="primary")
    audio_out = gr.Audio(label="خروجی", type="filepath")
    report_out = gr.Textbox(label="گزارش", lines=12)
    btn.click(generate_ui, inputs=[inp, emo, gender, intensity], outputs=[audio_out, report_out])
    gr.Examples(
        examples=[
            ["چه روز فوق‌العاده‌ای است، دلم از شادی می‌لرزد!", "شادی", "زنانه — Mana", 1.0],
            ["این رفتار کاملاً غیرقابل قبول است، خیلی عصبانی‌ام!", "خشم", "مردانه — Ganji", 1.0],
            ["دلم گرفته و حس می‌کنم همه چیز را از دست داده‌ام.", "اندوه", "زنانه — Mana", 1.05],
            ["گزارش جلسه امروز روی میز مدیر قرار گرفت.", "خنثی", "مردانه — Ganji", 0.0],
            ["باورم نمی‌شود! هیچ‌وقت چنین چیزی ندیده بودم!", "شگفتی", "زنانه — Mana", 1.0],
        ],
        inputs=[inp, emo, gender, intensity],
    )

demo.launch(debug=False, share=True)
""")

# =============================================================================
code(r"""
# ============================================================
# Cell 12: Final report + pass/fail
# ============================================================

import json
from datetime import datetime
from pathlib import Path
import numpy as np

PROJECT_DIR = Path("/content/PersianEmoTTS_Project") if Path("/content/PersianEmoTTS_Project").exists() else Path.cwd() / "PersianEmoTTS_Project"
REPORTS = PROJECT_DIR / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

final = {
    "created_at": datetime.now().isoformat(),
    "system": "PersianEmoTTS",
    "backbone": "Frozen Piper Mana (female) + Ganji (male)",
    "emotion_control": "Style vector + Praat PSOLA + Piper length/noise, control scale λ",
    "papers": [
        "TTS-CtrlNet (Jeong et al. 2025) — freeze backbone, add control",
        "StyleTTS 2 (Li et al. 2023) — style as latent",
        "EmoVoice (Yang et al. 2025) — emotion prompt + SER eval",
        "AutESD (Rahmati et al. 2026) — Persian 4-class SER protocol",
        "ShEMO (Mohamad Nezami et al. 2019) — public Persian emotion corpus",
        "PersianVox (2026) — Persian speech quality / prosody",
    ],
    "demo_text": DEMO_TEXT if "DEMO_TEXT" in dir() else None,
    "acoustic_gates_demo": gates if "gates" in dir() else None,
    "acoustic_gates_benchmark": gates2 if "gates2" in dir() else None,
    "mean_rtf": float(bench_df.rtf.mean()) if "bench_df" in dir() else None,
    "ser_uar": uar if "uar" in dir() else None,
    "ser_acc": acc if "acc" in dir() else None,
    "whisper_mean_cer": float(np.mean(cer_values)) if "cer_values" in dir() and cer_values else None,
    "n_benchmark_utts": int(len(bench_df)) if "bench_df" in dir() else None,
}

out = REPORTS / "final_report.json"
out.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

print("=" * 70)
print("PersianEmoTTS — FINAL")
print("=" * 70)
print(json.dumps(final, ensure_ascii=False, indent=2))
print()
ok_acoust = bool(final.get("acoustic_gates_demo", {}).get("passed")) and \
            bool(final.get("acoustic_gates_benchmark", {}).get("passed"))
print("Acoustic emotion gates :", "PASSED" if ok_acoust else "CHECK")
if final["mean_rtf"] is not None:
    print(f"Mean RTF               : {final['mean_rtf']:.4f}")
if final["ser_uar"] is not None:
    print(f"ShEMO-SVM UAR on TTS  : {final['ser_uar']:.3f}")
if final["whisper_mean_cer"] is not None:
    print(f"Whisper-small CER      : {final['whisper_mean_cer']:.3f}")
print("Report:", out)
print("=" * 70)
print("اگر سلول ۷ PASSED شده، احساس روی جملهٔ فارسی از نظر آکوستیک جدا شده")
print("و در سلول ۶ همان جمله را با ۶ احساس می‌توانید بشنوید.")
print("=" * 70)
""")

nb["cells"] = cells
outp = ROOT / "PersianEmoTTS.ipynb"
nbf.write(nb, outp)
print("wrote", outp, "cells", len(cells), "bytes", outp.stat().st_size)
