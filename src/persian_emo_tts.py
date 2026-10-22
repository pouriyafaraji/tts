#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PersianEmoTTS — emotion-controllable Persian text-to-speech engine.

Research path
-------------
* Backbone: frozen high-quality Persian Piper (Mana female / Ganji male),
  following the TTS-CtrlNet idea of keeping the pretrained TTS intact.
* Emotion control: a style vector (rate, F0, range, formant, energy, tilt,
  jitter, pause) interpolated with a control scale λ, then realised with
  Praat PSOLA (Parselmouth) — the same family of style / emotion controls
  studied in StyleTTS 2, EmoVoice and TTS-CtrlNet.
* Evaluation protocol: AutESD 4-class emotions + ShEMO extras, acoustic
  separability, SVM-SER (AutESD paper: KNN/SVM/CNN on MFCC), RTF, CER.

This module is imported by PersianEmoTTS.ipynb. It has no Colab-only
side effects at import time.
"""

from __future__ import annotations

import io
import json
import math
import re
import time
import unicodedata
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Emotion taxonomy  (AutESD: N/H/A/S   + ShEMO extras: F/W)
# ---------------------------------------------------------------------------

EMOTION_FA = {
    "neutral": "خنثی",
    "happy": "شادی",
    "angry": "خشم",
    "sad": "اندوه",
    "fear": "ترس",
    "surprise": "شگفتی",
}

EMOTION_ORDER = ("neutral", "happy", "angry", "sad", "fear", "surprise")
AUTESD_EMOTIONS = ("neutral", "happy", "angry", "sad")

# ShEMO filename codes
SHEMO_CODE = {"A": "angry", "H": "happy", "N": "neutral", "S": "sad", "F": "fear", "W": "surprise"}
SHEMO_INV = {v: k for k, v in SHEMO_CODE.items()}


# ---------------------------------------------------------------------------
# Style priors  (Banse & Scherer 1996; Juslin & Laukka 2003; AutESD 4-class)
# Values are *targets* at intensity=1.0, interpolated toward neutral by λ.
# ---------------------------------------------------------------------------

@dataclass
class EmotionStyle:
    length_scale: float = 1.0      # Piper duration (>1 slower)
    pitch_factor: float = 1.0      # multiply F0
    pitch_range: float = 1.0       # F0 variance
    formant: float = 1.0           # vocal-tract / smile ( >1 brighter )
    energy_db: float = 0.0         # RMS gain
    noise_scale: float = 0.667     # Piper generator noise
    noise_w: float = 0.80          # Piper duration noise
    pause: float = 1.0             # in-utterance pause stretch
    jitter: float = 0.0            # relative F0 micro-tremor
    tilt_db: float = 0.0           # high-shelf gain (brightness)
    saturation: float = 1.0        # tanh drive (tenseness)
    contour: str = "none"          # none | falling | rising | rise_fall | peak_early


STYLE_PRIORS: Dict[str, EmotionStyle] = {
    "neutral": EmotionStyle(),
    "happy": EmotionStyle(
        length_scale=0.88, pitch_factor=1.18, pitch_range=1.45, formant=1.08,
        energy_db=3.2, noise_scale=0.78, noise_w=0.95, pause=0.70,
        jitter=0.002, tilt_db=4.5, saturation=1.05, contour="rise_fall",
    ),
    "angry": EmotionStyle(
        length_scale=0.80, pitch_factor=1.12, pitch_range=1.30, formant=0.96,
        energy_db=6.0, noise_scale=0.95, noise_w=0.62, pause=0.48,
        jitter=0.005, tilt_db=2.4, saturation=1.32, contour="falling",
    ),
    "sad": EmotionStyle(
        length_scale=1.36, pitch_factor=0.80, pitch_range=0.58, formant=0.94,
        energy_db=-6.0, noise_scale=0.48, noise_w=1.05, pause=1.80,
        jitter=0.001, tilt_db=-5.5, saturation=0.88, contour="falling",
    ),
    "fear": EmotionStyle(
        length_scale=0.80, pitch_factor=1.20, pitch_range=1.45, formant=1.05,
        energy_db=1.2, noise_scale=0.88, noise_w=1.15, pause=0.65,
        jitter=0.014, tilt_db=1.4, saturation=1.08, contour="rising",
    ),
    "surprise": EmotionStyle(
        length_scale=0.86, pitch_factor=1.24, pitch_range=1.55, formant=1.06,
        energy_db=3.4, noise_scale=0.82, noise_w=1.00, pause=0.78,
        jitter=0.004, tilt_db=3.0, saturation=1.12, contour="peak_early",
    ),
}


def lerp_style(target: EmotionStyle, intensity: float) -> EmotionStyle:
    """Interpolate style toward the emotion prior with control scale λ ∈ [0, 1.25]."""
    n = STYLE_PRIORS["neutral"]
    t = float(np.clip(intensity, 0.0, 1.25))

    def mix(a, b):
        return a + (b - a) * t

    return EmotionStyle(
        length_scale=mix(n.length_scale, target.length_scale),
        pitch_factor=mix(n.pitch_factor, target.pitch_factor),
        pitch_range=mix(n.pitch_range, target.pitch_range),
        formant=mix(n.formant, target.formant),
        energy_db=mix(n.energy_db, target.energy_db),
        noise_scale=mix(n.noise_scale, target.noise_scale),
        noise_w=mix(n.noise_w, target.noise_w),
        pause=mix(n.pause, target.pause),
        jitter=mix(n.jitter, target.jitter),
        tilt_db=mix(n.tilt_db, target.tilt_db),
        saturation=mix(n.saturation, target.saturation),
        contour=target.contour if t > 0.15 else "none",
    )


# ---------------------------------------------------------------------------
# Persian frontend
# ---------------------------------------------------------------------------

_AR_YE, _FA_YE = "ي", "ی"
_AR_KAF, _FA_KAF = "ك", "ک"
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_ZWNJ = "\u200c"
_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_DIGIT_MAP = {p: str(i) for i, p in enumerate(_PERSIAN_DIGITS)}
_DIGIT_MAP.update({p: str(i) for i, p in enumerate(_ARABIC_DIGITS)})

_ONES = ["", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه"]
_TEENS = ["ده", "یازده", "دوازده", "سیزده", "چهارده", "پانزده",
          "شانزده", "هفده", "هجده", "نوزده"]
_TENS = ["", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"]
_HUNDREDS = ["", "صد", "دویست", "سیصد", "چهارصد", "پانصد",
             "ششصد", "هفتصد", "هشتصد", "نهصد"]
_SCALES = [(10 ** 9, "میلیارد"), (10 ** 6, "میلیون"), (10 ** 3, "هزار")]


def _three_digits(n: int) -> str:
    n = int(n)
    parts: List[str] = []
    h, rem = divmod(n, 100)
    if h:
        parts.append(_HUNDREDS[h])
    if 10 <= rem < 20:
        parts.append(_TEENS[rem - 10])
    else:
        t, o = divmod(rem, 10)
        if t:
            parts.append(_TENS[t])
        if o:
            parts.append(_ONES[o])
    return " و ".join([p for p in parts if p])


def number_to_fa(n: int) -> str:
    n = int(n)
    if n == 0:
        return "صفر"
    if n < 0:
        return "منفی " + number_to_fa(-n)
    parts: List[str] = []
    for scale, name in _SCALES:
        q, n = divmod(n, scale)
        if q:
            parts.append(_three_digits(q) + " " + name)
    if n:
        parts.append(_three_digits(n))
    return " و ".join(parts)


def _replace_numbers(text: str) -> str:
    def repl(m: re.Match) -> str:
        raw = m.group(0)
        if len(raw) > 12:
            return raw
        try:
            return number_to_fa(int(raw))
        except Exception:
            return raw
    return re.sub(r"\d+", repl, text)


def normalize_persian(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(_AR_YE, _FA_YE).replace(_AR_KAF, _FA_KAF)
    text = text.replace("ة", "ه").replace("ؤ", "و").replace("إ", "ا").replace("أ", "ا")
    text = text.replace("ۀ", "ه").replace("ء", "")
    text = text.translate(str.maketrans(_DIGIT_MAP))
    text = _DIACRITICS.sub("", text)
    text = text.replace("‌", _ZWNJ)
    text = re.sub(r"[ـ]+", "", text)
    text = re.sub(r"[\u200b\u200d\u2060]", "", text)
    text = re.sub(r"[“”«»]", '"', text)
    text = re.sub(r"[–—−]", "-", text)
    try:
        from hazm import Normalizer
        text = Normalizer().normalize(text)
    except Exception:
        pass
    text = _replace_numbers(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Text emotion planner  (keyword + punctuation; no extra model required)
# ---------------------------------------------------------------------------

_EMO_LEXICON: Dict[str, Tuple[str, float]] = {}


def _add_lex(emotion: str, weight: float, words: Sequence[str]) -> None:
    for w in words:
        _EMO_LEXICON[w] = (emotion, weight)


_add_lex("happy", 1.0, [
    "خوشحال", "شاد", "شادی", "خوشحالم", "عالی", "فوق‌العاده", "فوق العاده",
    "محشر", "عاشق", "دوستت دارم", "دوست دارم", "مبارک", "تبریک", "خنده",
    "می‌خندم", "ميخندم", "لبخند", "هیجان", "هیجان‌زده", "ذوق", "خوشبخت",
    "موفقیت", "برنده", "آفرین", "مرسی", "ممنون", "سپاس", "بهبه", "چه خوب",
    "لذت", "لذت‌بخش", "باحال", "با‌حال", "خنده‌دار", "جشن", "سورپرایز شیرین",
])
_add_lex("angry", 1.0, [
    "عصبانی", "خشم", "خشمگین", "خشمگینم", "نفرت", "متنفر", "لعنت", "احمق",
    "بی‌شرف", "بی شرف", "نابخشودنی", "تحمل نمی‌کنم", "بس است", "خفه",
    "فریاد", "داد می‌زنم", "خیانت", "دروغگو", "حقه‌باز", "حقه باز",
    "برخورد", "جنگی", "نابود", "غیرقابل قبول", "شرم‌آور", "شرم اور",
])
_add_lex("sad", 1.0, [
    "متأسف", "متاسف", "متاسفم", "متأسفم", "غمگین", "اندوه", "افسوس",
    "گریه", "می‌گریم", "میگریم", "دلم گرفته", "دلم شکست", "شکست خوردم",
    "تنهایی", "تنهام", "از دست دادم", "فوت", "مرگ", "غم", "محزون",
    "ناکام", "پشیمان", "افسرده", "نالان", "حسرت", "سوگ", "داغدار",
    "بیچاره", "محروم", "ناامید", "مایوس", "مایوسم",
])
_add_lex("fear", 1.0, [
    "می‌ترسم", "ميترسم", "ترسیدم", "ترس", "وحشت", "نگران", "نگرانم",
    "اضطراب", "هراس", "وحشتناک", "خطر", "فرار", "کمک", "نجات",
    "زلزله", "دزد", "تعقیب", "لرز", "قلبم", "وحشت‌زده",
])
_add_lex("surprise", 1.0, [
    "باورم نمی‌شود", "باورم نمیشود", "واقعاً", "واقعا", "عجیب", "شگفت",
    "شوکه", "نمی‌دانستم", "نمیدانستم", "چه خبر", "وای", "اوپس",
    "غیرمنتظره", "ناگهان", "یک‌دفعه", "یکدفعه",
])


def detect_text_emotion(text: str) -> Tuple[str, float, Dict[str, float]]:
    """Return (emotion, confidence, per-class scores)."""
    t = normalize_persian(text)
    scores = {e: 0.0 for e in EMOTION_ORDER}
    for w, (emo, wt) in _EMO_LEXICON.items():
        if w in t:
            scores[emo] += wt
    bang = t.count("!") + t.count("！")
    q = t.count("؟") + t.count("?")
    if bang >= 2:
        scores["angry"] += 0.6
        scores["surprise"] += 0.4
    elif bang == 1:
        scores["happy"] += 0.25
        scores["angry"] += 0.25
    if q >= 2:
        scores["surprise"] += 0.4
        scores["fear"] += 0.2
    best = max(scores, key=scores.get)
    total = sum(scores.values())
    if total < 0.4:
        return "neutral", 0.55, scores
    conf = scores[best] / (total + 1e-6)
    return best, float(np.clip(0.5 + 0.5 * conf, 0.5, 0.98)), scores


# ---------------------------------------------------------------------------
# Acoustic features  (AutESD-style: MFCC + F0 + energy)
# ---------------------------------------------------------------------------

def _safe_librosa():
    import librosa
    return librosa


def extract_acoustic_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.float32).flatten()
    if y.size < 16:
        return {k: 0.0 for k in (
            "duration", "rms", "rms_std", "f0_mean", "f0_std", "f0_min", "f0_max",
            "f0_range", "zcr", "centroid", "tilt", "speech_rate_proxy",
            *{f"mfcc_{i}_{stat}" for i in range(13) for stat in ("mean", "std")}
        )}
    duration = y.size / float(sr)
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)) + 1e-9)
    y = y / peak  # peak-norm only for F0 / MFCC, not for loudness
    hop = 512
    frame = 1024
    n_frames = max(1, (y.size - frame) // hop + 1)
    rms_frames = []
    for i in range(n_frames):
        sl = y[i * hop: i * hop + frame]
        rms_frames.append(float(np.sqrt(np.mean(sl ** 2) + 1e-12)))
    rms_std = float(np.std(rms_frames))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(y)))))
    feat: Dict[str, float] = {
        "duration": duration,
        "rms": rms,
        "rms_std": rms_std,
        "zcr": zcr,
        "speech_rate_proxy": 1.0 / max(duration, 1e-3),
    }
    try:
        librosa = _safe_librosa()
        f0, _, _ = librosa.pyin(
            y, fmin=60, fmax=500, sr=sr, frame_length=2048, hop_length=hop,
        )
        f0v = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        if f0v.size >= 3:
            feat["f0_mean"] = float(np.mean(f0v))
            feat["f0_std"] = float(np.std(f0v))
            feat["f0_min"] = float(np.min(f0v))
            feat["f0_max"] = float(np.max(f0v))
            feat["f0_range"] = feat["f0_max"] - feat["f0_min"]
        else:
            feat.update(f0_mean=0.0, f0_std=0.0, f0_min=0.0, f0_max=0.0, f0_range=0.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
        for i in range(13):
            feat[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
            feat[f"mfcc_{i}_std"] = float(np.std(mfcc[i]))
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)
        feat["centroid"] = float(np.mean(cent))
        spec = np.abs(librosa.stft(y, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=sr)
        mag = np.mean(spec, axis=1) + 1e-12
        logf = np.log(np.maximum(freqs, 1.0))
        logm = np.log(mag)
        A = np.vstack([logf, np.ones_like(logf)]).T
        slope, _ = np.linalg.lstsq(A, logm, rcond=None)[0]
        feat["tilt"] = float(slope)
    except Exception:
        feat.update(f0_mean=0.0, f0_std=0.0, f0_min=0.0, f0_max=0.0, f0_range=0.0,
                    centroid=0.0, tilt=0.0)
        for i in range(13):
            feat.setdefault(f"mfcc_{i}_mean", 0.0)
            feat.setdefault(f"mfcc_{i}_std", 0.0)
    return feat


FEATURE_KEYS = [
    "duration", "rms", "rms_std", "f0_mean", "f0_std", "f0_range",
    "zcr", "centroid", "tilt", "speech_rate_proxy",
] + [f"mfcc_{i}_{s}" for i in range(13) for s in ("mean", "std")]


def features_to_vector(feat: Dict[str, float]) -> np.ndarray:
    return np.array([float(feat.get(k, 0.0)) for k in FEATURE_KEYS], dtype=np.float32)


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

def _high_shelf(y: np.ndarray, sr: int, gain_db: float, cutoff: float = 2800.0) -> np.ndarray:
    if abs(gain_db) < 0.15:
        return y
    try:
        from scipy.signal import iirfilter, sosfilt
        sos = iirfilter(2, cutoff / (sr / 2.0), btype="high", ftype="butter", output="sos")
        high = sosfilt(sos, y)
        g = 10.0 ** (gain_db / 20.0)
        out = y + (g - 1.0) * high.astype(np.float32)
        return out.astype(np.float32)
    except Exception:
        return y


def _stretch_pauses(y: np.ndarray, sr: int, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 0.05:
        return y
    frame = max(1, int(0.025 * sr))
    hop = max(1, int(0.010 * sr))
    energy = []
    for i in range(0, max(1, y.size - frame), hop):
        energy.append(float(np.mean(y[i:i + frame] ** 2)))
    if not energy:
        return y
    thr = 0.15 * float(np.median(energy) + 1e-12)
    out = []
    i = 0
    n = y.size
    silent = False
    start = 0
    while i < n:
        fi = min(len(energy) - 1, i // hop)
        is_sil = energy[fi] < thr
        if is_sil != silent:
            seg = y[start:i]
            if silent and factor > 1.0:
                # repeat silence
                extra = int(seg.size * (factor - 1.0))
                out.append(seg)
                if extra > 0:
                    out.append(np.zeros(extra, dtype=np.float32))
            elif silent and factor < 1.0:
                keep = max(int(seg.size * factor), int(0.02 * sr))
                out.append(seg[:keep])
            else:
                out.append(seg)
            silent = is_sil
            start = i
        i += hop
    out.append(y[start:])
    return np.concatenate(out) if out else y


def _apply_saturation(y: np.ndarray, drive: float) -> np.ndarray:
    if abs(drive - 1.0) < 0.03:
        return y
    d = float(np.clip(drive, 0.7, 1.8))
    return np.tanh(y * d).astype(np.float32) / math.tanh(d)


def _peak_normalize(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(y)) + 1e-9)
    return (y * (peak / m)).astype(np.float32)


def _limit(y: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    """Scale down only if the signal would clip — keeps loud/soft contrast."""
    m = float(np.max(np.abs(y)) + 1e-9)
    if m <= ceiling:
        return y.astype(np.float32)
    return (y * (ceiling / m)).astype(np.float32)


# ---------------------------------------------------------------------------
# Praat / Parselmouth emotion renderer
# ---------------------------------------------------------------------------

def _contour_multiplier(rel: float, kind: str) -> float:
    rel = float(np.clip(rel, 0.0, 1.0))
    if kind == "falling":
        return 1.10 - 0.20 * rel
    if kind == "rising":
        return 0.92 + 0.20 * rel
    if kind == "rise_fall":
        return 1.0 + 0.14 * math.sin(math.pi * rel)
    if kind == "peak_early":
        # surprise: jump then settle
        if rel < 0.18:
            return 1.18
        return 1.18 - 0.22 * ((rel - 0.18) / 0.82)
    return 1.0


def render_emotion_psola(y: np.ndarray, sr: int, style: EmotionStyle) -> np.ndarray:
    """Apply F0 / formant / range / contour with Praat PSOLA; fallback to numpy."""
    y = np.asarray(y, dtype=np.float32).flatten()
    try:
        import parselmouth
        from parselmouth.praat import call
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        # Formant + pitch-range via "Change gender" (keeps identity reasonably)
        formant = float(np.clip(style.formant, 0.85, 1.20))
        prange = float(np.clip(style.pitch_range, 0.40, 1.80))
        snd = call(snd, "Change gender", 75, 600, formant, 0.0, prange, 1.0)

        manipulation = call(snd, "To Manipulation", 0.01, 75, 600)
        pitch_tier = call(manipulation, "Extract pitch tier")
        n_pts = int(call(pitch_tier, "Get number of points"))
        xmin = float(snd.xmin)
        xmax = float(snd.xmax)
        dur = max(xmax - xmin, 1e-4)
        rng = np.random.default_rng(0)
        new_tier = call("Create PitchTier", "emo", xmin, xmax)
        pf = float(style.pitch_factor)
        for i in range(1, n_pts + 1):
            t = float(call(pitch_tier, "Get time from index", i))
            f = float(call(pitch_tier, "Get value at index", i))
            rel = (t - xmin) / dur
            f2 = f * pf * _contour_multiplier(rel, style.contour)
            if style.jitter > 0:
                f2 *= 1.0 + float(rng.normal(0.0, style.jitter))
            f2 = float(np.clip(f2, 50.0, 600.0))
            call(new_tier, "Add point", t, f2)
        call([new_tier, manipulation], "Replace pitch tier")
        out = call(manipulation, "Get resynthesis (overlap-add)")
        y2 = np.array(out.values, dtype=np.float32).flatten()
        return y2
    except Exception:
        # Fallback: global pitch shift (librosa, then naive resample)
        try:
            import librosa
            n_steps = 12.0 * math.log2(max(style.pitch_factor, 1e-3))
            y2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            return np.asarray(y2, dtype=np.float32)
        except Exception:
            factor = float(max(style.pitch_factor, 0.5))
            n = y.size
            t_old = np.linspace(0.0, 1.0, n)
            n2 = max(16, int(round(n / factor)))
            t_new = np.linspace(0.0, 1.0, n2)
            y2 = np.interp(t_new, t_old, y).astype(np.float32)
            # restore original duration
            t_back = np.linspace(0.0, 1.0, n)
            t_y2 = np.linspace(0.0, 1.0, y2.size)
            return np.interp(t_back, t_y2, y2).astype(np.float32)


def apply_emotion_dsp(y: np.ndarray, sr: int, style: EmotionStyle) -> np.ndarray:
    y = render_emotion_psola(y, sr, style)
    y = _stretch_pauses(y, sr, style.pause)
    y = _high_shelf(y, sr, style.tilt_db)
    y = _apply_saturation(y, style.saturation)
    # relative loudness — do NOT peak-normalize, or anger/sadness collapse
    rms0 = float(np.sqrt(np.mean(y ** 2)) + 1e-9)
    target = rms0 * (10.0 ** (style.energy_db / 20.0))
    y = y * (target / rms0)
    y = _limit(y, 0.98)
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Piper backbone
# ---------------------------------------------------------------------------

class PiperBackbone:
    def __init__(self) -> None:
        self.voices: Dict[str, Any] = {}
        self.paths: Dict[str, Path] = {}
        self.sample_rate = 22050

    def load(self, gender: str, onnx_path: Path, json_path: Optional[Path] = None) -> None:
        try:
            from piper import PiperVoice
        except Exception:
            from piper.voice import PiperVoice  # type: ignore
        kwargs = {}
        if json_path is not None and Path(json_path).exists():
            try:
                self.voices[gender] = PiperVoice.load(
                    str(onnx_path), config_path=str(json_path), use_cuda=False
                )
            except TypeError:
                try:
                    self.voices[gender] = PiperVoice.load(str(onnx_path), config_path=str(json_path))
                except TypeError:
                    self.voices[gender] = PiperVoice.load(str(onnx_path))
        else:
            try:
                self.voices[gender] = PiperVoice.load(str(onnx_path), use_cuda=False)
            except TypeError:
                self.voices[gender] = PiperVoice.load(str(onnx_path))
        self.paths[gender] = Path(onnx_path)
        v = self.voices[gender]
        sr = getattr(getattr(v, "config", None), "sample_rate", None)
        if sr:
            self.sample_rate = int(sr)

    def synthesize(
        self,
        text: str,
        gender: str = "female",
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        volume: float = 1.0,
        phonemes: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        if gender not in self.voices:
            gender = next(iter(self.voices))
        voice = self.voices[gender]
        payload = f"[[ {phonemes} ]]" if phonemes else text
        sr = int(getattr(getattr(voice, "config", None), "sample_rate", self.sample_rate) or 22050)
        audio = None
        try:
            from piper import SynthesisConfig
            cfg = SynthesisConfig(
                length_scale=float(length_scale),
                noise_scale=float(noise_scale),
                noise_w_scale=float(noise_w),
                volume=float(volume),
                normalize_audio=False,
            )
            chunks = list(voice.synthesize(payload, syn_config=cfg))
            audio = _chunks_to_float(chunks)
        except TypeError:
            try:
                chunks = list(voice.synthesize(
                    payload,
                    length_scale=float(length_scale),
                    noise_scale=float(noise_scale),
                    noise_w=float(noise_w),
                ))
                audio = _chunks_to_float(chunks)
            except Exception:
                audio = None
        except Exception:
            audio = None
        if audio is None:
            # last-resort: write wav via older API
            import tempfile, wave
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            with wave.open(tmp.name, "wb") as wf:
                voice.synthesize(payload, wf)
            import soundfile as sf
            audio, sr = sf.read(tmp.name, dtype="float32")
            Path(tmp.name).unlink(missing_ok=True)
        audio = np.asarray(audio, dtype=np.float32).flatten()
        return audio, sr


def _chunks_to_float(chunks: Sequence[Any]) -> np.ndarray:
    parts = []
    for c in chunks:
        if hasattr(c, "audio_float_array"):
            parts.append(np.asarray(c.audio_float_array, dtype=np.float32).flatten())
        elif hasattr(c, "audio_int16_bytes"):
            a = np.frombuffer(c.audio_int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            parts.append(a)
        elif isinstance(c, np.ndarray):
            parts.append(np.asarray(c, dtype=np.float32).flatten())
    if not parts:
        return np.zeros(1600, dtype=np.float32)
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Optional Homo-GE2PE G2P  (from the professor's FinalProject.ipynb)
# ---------------------------------------------------------------------------

PHONEME_MAPPING = {
    "/": "a", "a": "ɑ", "e": "e", "o": "o", "i": "i", "u": "u",
    "@": "ʔ", "?": "ʔ", "$": "ʃ", "c": "tʃ", ";": "ʒ", "g": "ɡ",
    "y": "j", "q": "qq", "1": "",
}
PUNCTUATION = ",."


def homo_to_piper(homo_phonemic_text: str, original_text: str) -> str:
    homo_phonemic_text = re.sub(r"(?<!\w)@ali\b", "@/li", homo_phonemic_text)
    homo_phonemic_text = re.sub(r"h\$il", "hsil", homo_phonemic_text)
    orig_tokens = re.findall(r"\w+|[^\w\s]", original_text, flags=re.UNICODE)
    homo_tokens = homo_phonemic_text.split()
    rebuilt: List[str] = []
    h = 0
    for tok in orig_tokens:
        if re.fullmatch(r"[،,.;؛:!?؟]", tok):
            if rebuilt:
                if tok in "،,؛;:":
                    rebuilt[-1] += ","
                elif tok in ".!?؟":
                    rebuilt[-1] = rebuilt[-1].replace("1", "")
                    if rebuilt[-1].endswith("e"):
                        rebuilt[-1] = rebuilt[-1][:-1]
                    rebuilt[-1] += "."
            continue
        if h >= len(homo_tokens):
            break
        rebuilt.append(homo_tokens[h])
        h += 1
    homo_phonemic_text = " ".join(rebuilt).replace("1", "")
    translated = []
    has_zhe = "ژ" in original_text
    for ch in homo_phonemic_text:
        if ch in PUNCTUATION:
            translated.append(ch)
            continue
        if ch == "j":
            translated.append("ʒ" if has_zhe else "dʒ")
            continue
        translated.append(PHONEME_MAPPING.get(ch, ch))
    return "".join(translated)


class OptionalG2P:
    def __init__(self) -> None:
        self.model = None

    def load(self, model_path: str) -> None:
        from GE2PE import GE2PE
        self.model = GE2PE(model_path=model_path)

    def __call__(self, text: str) -> Optional[str]:
        if self.model is None:
            return None
        try:
            homo = self.model.generate([text], use_rules=True)[0]
            return homo_to_piper(homo, text)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# High-level synthesizer
# ---------------------------------------------------------------------------

@dataclass
class SynthResult:
    audio: np.ndarray
    sr: int
    emotion: str
    intensity: float
    gender: str
    text: str
    normalized: str
    rtf: float
    synth_seconds: float
    duration: float
    style: Dict[str, Any]
    phonemes: Optional[str] = None
    text_emotion: Optional[str] = None
    text_emotion_conf: Optional[float] = None


class PersianEmoTTS:
    def __init__(self) -> None:
        self.backbone = PiperBackbone()
        self.g2p = OptionalG2P()
        self.use_g2p = False

    def synthesize(
        self,
        text: str,
        emotion: str = "auto",
        intensity: float = 0.90,
        gender: str = "female",
        use_g2p: Optional[bool] = None,
    ) -> SynthResult:
        t0 = time.time()
        normalized = normalize_persian(text)
        text_emo, text_conf, _ = detect_text_emotion(normalized)
        if emotion in (None, "", "auto", "خودکار"):
            emotion = text_emo
        emotion = emotion.lower().strip()
        aliases = {
            "خنثی": "neutral", "شادی": "happy", "خوشحال": "happy", "خشم": "angry",
            "عصبانی": "angry", "اندوه": "sad", "غمگین": "sad", "ترس": "fear",
            "شگفتی": "surprise", "تعجب": "surprise",
        }
        emotion = aliases.get(emotion, emotion)
        if emotion not in STYLE_PRIORS:
            emotion = "neutral"
        style = lerp_style(STYLE_PRIORS[emotion], intensity)
        phonemes = None
        if (use_g2p if use_g2p is not None else self.use_g2p):
            phonemes = self.g2p(normalized)
        audio, sr = self.backbone.synthesize(
            normalized,
            gender=gender,
            length_scale=style.length_scale,
            noise_scale=style.noise_scale,
            noise_w=style.noise_w,
            volume=1.0,
            phonemes=phonemes,
        )
        if emotion != "neutral" and intensity > 0.02:
            audio = apply_emotion_dsp(audio, sr, style)
        else:
            audio = _peak_normalize(audio, 0.95)
        elapsed = time.time() - t0
        dur = float(audio.size / max(sr, 1))
        rtf = elapsed / max(dur, 1e-6)
        return SynthResult(
            audio=audio, sr=sr, emotion=emotion, intensity=float(intensity),
            gender=gender, text=text, normalized=normalized, rtf=rtf,
            synth_seconds=elapsed, duration=dur,
            style=asdict(style), phonemes=phonemes,
            text_emotion=text_emo, text_emotion_conf=text_conf,
        )


def save_wav(path: Path, audio: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    sf.write(str(path), np.asarray(audio, dtype=np.float32), int(sr))
    return path


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def acoustic_gates(by_emotion: Dict[str, List[Dict[str, float]]]) -> Dict[str, Any]:
    """Check that generated emotions separate the way speech science predicts."""
    def mean(emo, key):
        vals = [f.get(key, 0.0) for f in by_emotion.get(emo, [])]
        if key.startswith("f0"):
            vals = [v for v in vals if v and v > 0]
        if not vals:
            return None
        return float(np.mean(vals))

    f0_h, f0_s = mean("happy", "f0_mean"), mean("sad", "f0_mean")
    f0_a = mean("angry", "f0_mean")
    dur_s, dur_a = mean("sad", "duration"), mean("angry", "duration")
    e_a, e_s = mean("angry", "rms"), mean("sad", "rms")
    e_h = mean("happy", "rms")

    checks = {
        "f0_happy_gt_sad": bool(f0_h and f0_s and f0_h > f0_s),
        "f0_angry_gt_sad": bool(f0_a and f0_s and f0_a > f0_s),
        "dur_sad_gt_angry": bool(dur_s and dur_a and dur_s > dur_a),
        "energy_angry_gt_sad": bool(e_a and e_s and e_a > e_s),
        "energy_happy_gt_sad": bool(e_h and e_s and e_h > e_s),
    }
    checks["passed"] = all(checks.values())
    checks["means"] = {
        "f0_happy": f0_h, "f0_sad": f0_s, "f0_angry": f0_a,
        "dur_sad": dur_s, "dur_angry": dur_a,
        "rms_angry": e_a, "rms_sad": e_s, "rms_happy": e_h,
    }
    return checks


def uar(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    recalls = []
    for lab in labels:
        idx = [i for i, t in enumerate(y_true) if t == lab]
        if not idx:
            continue
        ok = sum(1 for i in idx if y_pred[i] == lab)
        recalls.append(ok / len(idx))
    return float(np.mean(recalls)) if recalls else 0.0


NEUTRAL_EVAL_SENTENCES = [
    "فردا ساعت سه در میدان اصلی شهر منتظر شما می‌مانم.",
    "گزارش جلسه امروز روی میز مدیر قرار گرفت.",
    "هوا کمی ابری است و باد می‌وزد.",
    "کتاب را روی میز کنار پنجره گذاشتم.",
    "قطار رأس ساعت نه از ایستگاه حرکت می‌کند.",
    "این اتاق سه پنجره بزرگ رو به حیاط دارد.",
]

CONGRUENT_SENTENCES = {
    "happy": "چه روز فوق‌العاده‌ای است، دلم از شادی می‌لرزد!",
    "angry": "این رفتار کاملاً غیرقابل قبول است، خیلی عصبانی‌ام!",
    "sad": "دلم گرفته و حس می‌کنم همه چیز را از دست داده‌ام.",
    "fear": "می‌ترسم، این صدا از تاریکی می‌آید، کمک کنید.",
    "surprise": "باورم نمی‌شود! هیچ‌وقت چنین چیزی ندیده بودم!",
    "neutral": "امروز جلسه ساعت ده در طبقه دوم برگزار می‌شود.",
}
