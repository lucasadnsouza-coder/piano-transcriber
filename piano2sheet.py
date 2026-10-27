#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
piano2sheet - de um video de piano no YouTube para partitura + MIDI (Synthesia).

Pipeline:
  1. yt-dlp baixa o audio; ffmpeg converte para WAV 44.1 kHz estereo.
  2. Transkun (semi-CRF, estado da arte em transcricao de piano) detecta notas,
     velocity e pedal de sustentacao.
  3. Beat This! detecta tempos e tempos fortes -> mapa de andamento e compasso.
  4. Busca em feixe separa mao direita / mao esquerda; Viterbi escolhe, tempo a
     tempo, a grade ritmica (colcheia, semicolcheia, tercina, fusa...).
  5. Escreve MIDI para Synthesia (uma faixa por mao, mapa de andamento que segue
     o interprete), MusicXML (music21) e PDF (MuseScore, ou Verovio como reserva).

Uso:
  python piano2sheet.py "https://www.youtube.com/watch?v=..."
  python piano2sheet.py gravacao.mp3 --time-signature 3/4
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

PPQ = 480
CACHE_VERSION = 1
HAND_NAMES = ("Right Hand", "Left Hand")  # nomes que o Synthesia entende como maos
RIGHT, LEFT = 0, 1


@dataclass
class Note:
    start: float  # segundos
    end: float  # segundos (soltura da tecla, sem pedal)
    pitch: int
    velocity: int
    hand: int = RIGHT
    q_start: Fraction = Fraction(0)  # posicao quantizada, em tempos (beats)
    q_end: Fraction = Fraction(0)


@dataclass
class Meter:
    num: int
    den: int
    bpb: int  # tempos (beats) por compasso
    beat_ql: Fraction  # duracao de um tempo em seminimas

    @property
    def compound(self) -> bool:
        return self.den == 8 and self.num % 3 == 0 and self.num > 3

    @staticmethod
    def parse(text: str) -> "Meter":
        m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", text)
        if not m:
            raise SystemExit(f"Formula de compasso invalida: {text!r} (use por ex. 3/4 ou 6/8)")
        num, den = int(m.group(1)), int(m.group(2))
        if den not in (2, 4, 8, 16):
            raise SystemExit("Denominador do compasso deve ser 2, 4, 8 ou 16.")
        if den == 8 and num % 3 == 0 and num > 3:
            return Meter(num, den, num // 3, Fraction(3, 2))
        return Meter(num, den, num, Fraction(4, den))

    @staticmethod
    def simple(bpb: int) -> "Meter":
        return Meter(bpb, 4, bpb, Fraction(1))

    def to_compound(self) -> "Meter":
        return Meter(self.bpb * 3, 8, self.bpb, Fraction(3, 2))

    def __str__(self) -> str:
        return f"{self.num}/{self.den}"


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_time(text: str | None) -> float | None:
    if text is None:
        return None
    total = 0.0
    for part in str(text).split(":"):
        total = total * 60 + float(part)
    return total


def safe_name(text: str, maxlen: int = 80) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text)
    text = re.sub(r"\s+", " ", text).strip().strip(".")
    return text[:maxlen].rstrip() or "partitura"


# ----------------------------------------------------------------- audio


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover
        raise SystemExit("ffmpeg nao encontrado. Instale com: pip install imageio-ffmpeg") from exc


def ensure_js_runtime() -> None:
    """O yt-dlp precisa de um runtime JavaScript (Deno) para extrair do YouTube."""
    if shutil.which("deno"):
        return
    try:
        import deno

        os.environ["PATH"] = str(Path(deno.find_deno_bin()).parent) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def is_url(text: str) -> bool:
    return re.match(r"^https?://", text, re.I) is not None


def youtube_info(url: str) -> dict:
    import yt_dlp

    ensure_js_runtime()
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}) as ydl:
        return ydl.extract_info(url, download=False)


def download_audio(url: str, dest: Path) -> Path:
    import yt_dlp

    ensure_js_runtime()
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(dest / "fonte.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    for item in info.get("requested_downloads") or []:
        if item.get("filepath") and Path(item["filepath"]).exists():
            return Path(item["filepath"])
    found = sorted(dest.glob("fonte.*"))
    if not found:
        raise SystemExit("Falha ao baixar o audio do video.")
    return found[0]


def to_wav(src: Path, dst: Path, start: float | None, end: float | None) -> None:
    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-i", str(src), "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio, sr


# ------------------------------------------------------ redes neurais


def pick_device(arg: str) -> str:
    import torch

    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def transcribe_audio(audio: np.ndarray, sr: int, device: str) -> tuple[list[Note], list[tuple[float, float]]]:
    """Transkun 2.0: notas (com offsets e velocity) + pedal de sustentacao."""
    from importlib import resources

    import moduleconf
    import torch

    pre = resources.files("transkun") / "pretrained"
    confman = moduleconf.parseFromFile(str(pre / "2.0.conf"))
    model = confman["Model"].module.TransKun(conf=confman["Model"].config).to(device)
    ckpt = torch.load(str(pre / "2.0.pt"), map_location=device, weights_only=False)
    state = ckpt["best_state_dict"] if "best_state_dict" in ckpt else ckpt["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    if sr != model.fs:
        import soxr

        audio = soxr.resample(audio, sr, model.fs)
    x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).to(device)
    with torch.inference_mode():
        events = model.transcribe(x)
    notes, pedals = [], []
    for ev in events:
        if ev.pitch > 0:
            notes.append(Note(float(ev.start), float(ev.end), int(ev.pitch), int(ev.velocity)))
        elif ev.pitch == -64:
            pedals.append((float(ev.start), float(ev.end)))
    notes.sort(key=lambda n: (n.start, n.pitch))
    pedals.sort()
    return notes, pedals


def track_beats(audio: np.ndarray, sr: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Beat This! (CPJKU, 2024): tempos e tempos fortes em segundos."""
    from beat_this.inference import Audio2Beats

    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    tracker = Audio2Beats(checkpoint_path="final0", device=device, dbn=False)
    beats, downbeats = tracker(mono.astype(np.float32), sr)
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


# ------------------------------------------------- separacao das maos


SPAN_COST = 0.12  # por semitom de abertura simultanea acima de uma quinta
POSITION_COST = 0.15  # por semitom alem de POSITION_RANGE coberto pela mao em pouco tempo (deslocar a mao)
POSITION_RANGE = 12
HOLD_COST = 0.2  # atacar nota nova enquanto a mesma mao segura outras
MEMORY_COST = 0.8  # trocar de mao uma tecla tocada ha pouco (padroes repetidos ficam coerentes)
MEMORY_SECONDS = 3.0
RECENT_SECONDS = 0.8


def _memory_cost(ps, hand, memory, t):
    cost = 0.0
    for p in ps:
        for mp, mh, mt in reversed(memory):
            if mp == p:
                if mh != hand and t - mt < MEMORY_SECONDS:
                    cost += MEMORY_COST
                break
    return cost


def _hand_cost(ps, held, recent, center, last_t, t, left):
    if not ps:
        return 0.0, center, last_t
    allp = ps + held
    span = max(allp) - min(allp)
    cost = SPAN_COST * max(0, span - 7)  # mao relaxada cobre ate uma quinta
    if span > 12:  # alem de uma oitava fica caro; 10a ou mais quase impossivel
        cost += 0.8 * (span - 12) ** 2
    if len(allp) > 5:
        cost += 25.0 * (len(allp) - 5)
    # posicao da mao: arpejo dentro de uma 6a nao e deslocamento
    win = allp + [p for p, pt in recent if t - pt < RECENT_SECONDS]
    cost += POSITION_COST * max(0, max(win) - min(win) - POSITION_RANGE)
    cost += HOLD_COST * len(held) * len(ps)
    mean = sum(ps) / len(ps)
    dist = abs(mean - center)
    # saltos grandes em pouco tempo custam mais; com a mao parada ha tempo, quase nada
    cost += 0.03 * dist + 0.35 * max(0.0, dist - 7) * math.exp(-(t - last_t) / 0.5)
    if left:
        cost += 0.05 * max(0.0, mean - 62) + 0.5 * max(0.0, mean - 74)
    else:
        cost += 0.05 * max(0.0, 58 - mean) + 0.5 * max(0.0, 46 - mean)
    return cost, 0.4 * center + 0.6 * mean, t


def split_hands(notes: list[Note], chord_window: float = 0.035, beam_width: int = 48) -> None:
    """Atribui cada nota a uma mao com busca em feixe sobre os acordes.

    Em cada acorde as notas sao ordenadas por altura e um ponto de corte decide o
    que vai para cada mao. O custo penaliza aberturas impossiveis, mais de 5 notas
    por mao, saltos rapidos, cruzamento das maos (inclusive com notas ainda
    seguradas) e registro improvavel.
    """
    if not notes:
        return
    order = sorted(range(len(notes)), key=lambda i: (notes[i].start, notes[i].pitch))
    groups: list[list[int]] = []
    for i in order:
        if groups and notes[i].start - notes[groups[-1][0]].start <= chord_window:
            groups[-1].append(i)
        else:
            groups.append([i])

    # estado: (custo, centro_esq, centro_dir, t_esq, t_dir, seguradas_esq, seguradas_dir, caminho,
    #          memoria, recentes_esq, recentes_dir)
    beam = [(0.0, 50.0, 71.0, -1e9, -1e9, (), (), None, (), (), ())]
    for g in groups:
        g.sort(key=lambda i: notes[i].pitch)
        t = min(notes[i].start for i in g)
        ps = [notes[i].pitch for i in g]
        ends = [notes[i].end for i in g]
        cand: dict = {}
        for cost, lc, rc, lt, rt, la, ra, path, mem, lrec, rrec in beam:
            la = tuple(a for a in la if a[1] > t + 0.05)
            ra = tuple(a for a in ra if a[1] > t + 0.05)
            lrec = tuple(a for a in lrec if t - a[1] < RECENT_SECONDS)
            rrec = tuple(a for a in rrec if t - a[1] < RECENT_SECONDS)
            lheld = [a[0] for a in la]
            rheld = [a[0] for a in ra]
            for k in range(len(ps) + 1):
                cl, lc2, lt2 = _hand_cost(ps[:k], lheld, lrec, lc, lt, t, True)
                cr, rc2, rt2 = _hand_cost(ps[k:], rheld, rrec, rc, rt, t, False)
                c = cost + cl + cr + _memory_cost(ps[:k], LEFT, mem, t) + _memory_cost(ps[k:], RIGHT, mem, t)
                if lc2 > rc2 - 2:
                    c += 3.0 + (lc2 - rc2 + 2)
                if k and rheld and ps[k - 1] > min(rheld):
                    c += 4.0
                if k < len(ps) and lheld and ps[k] < max(lheld):
                    c += 4.0
                nla = la + tuple(zip(ps[:k], ends[:k]))
                nra = ra + tuple(zip(ps[k:], ends[k:]))
                key = (round(lc2), round(rc2), tuple(sorted(a[0] for a in nla)), tuple(sorted(a[0] for a in nra)))
                if key not in cand or cand[key][0] > c:
                    nmem = (mem + tuple((p, LEFT, t) for p in ps[:k]) + tuple((p, RIGHT, t) for p in ps[k:]))[-24:]
                    cand[key] = (c, lc2, rc2, lt2, rt2, nla, nra, (path, k), nmem,
                                 lrec + tuple((p, t) for p in ps[:k]), rrec + tuple((p, t) for p in ps[k:]))
        beam = heapq.nsmallest(beam_width, cand.values(), key=lambda s: s[0])

    choices = []
    path = beam[0][7]
    while path is not None:
        path, k = path
        choices.append(k)
    choices.reverse()
    for g, k in zip(groups, choices):
        for j, i in enumerate(g):
            notes[i].hand = LEFT if j < k else RIGHT


def split_hands_fixed(notes: list[Note], split_point: int) -> None:
    for n in notes:
        n.hand = LEFT if n.pitch < split_point else RIGHT


# ------------------------------------------------- tempos e compasso


def _running_median(d: np.ndarray, half: int = 8) -> np.ndarray:
    return np.array([np.median(d[max(0, i - half): i + half + 1]) for i in range(len(d))])


def clean_beats(beats: np.ndarray) -> np.ndarray:
    """Preenche tempos perdidos e remove tempos espurios do detector."""
    b = np.unique(np.asarray(beats, dtype=float))
    if len(b) < 8:
        return b
    for _ in range(2):
        d = np.diff(b)
        med = _running_median(d)
        out = [b[0]]
        for i, di in enumerate(d):
            ratio = di / med[i]
            if ratio > 1.6:
                k = int(round(ratio))
                out.extend(b[i] + di * j / k for j in range(1, k))
            out.append(b[i + 1])
        b = np.array(out)
        med = _running_median(np.diff(b))
        keep = [b[0]]
        for i in range(1, len(b)):
            if b[i] - keep[-1] < 0.55 * med[min(i - 1, len(med) - 1)]:
                continue
            keep.append(b[i])
        b = np.array(keep)
    return b


def estimate_meter(beats: np.ndarray, downbeats: np.ndarray, bpb: int | None = None) -> tuple[int, int]:
    """Devolve (tempos por compasso, fase) a partir dos tempos fortes detectados."""
    if len(beats) < 8 or len(downbeats) < 2:
        return bpb or 4, 0
    med = float(np.median(np.diff(beats)))
    idx = []
    for d in downbeats:
        i = int(np.argmin(np.abs(beats - d)))
        if abs(beats[i] - d) < 0.3 * med:
            idx.append(i)
    idx = sorted(set(idx))
    if bpb is None:
        cnt: Counter = Counter()
        for d in np.diff(idx):
            if d in (2, 3, 4):
                cnt[int(d)] += 1.0
            elif d in (6, 8):  # um tempo forte perdido
                cnt[int(d) // 2] += 0.5
        bpb = cnt.most_common(1)[0][0] if cnt else 4
    if not idx:
        return bpb, 0
    phase = Counter(i % bpb for i in idx).most_common(1)[0][0]
    return bpb, phase


def harmonic_bar_check(notes: list[Note], beats: np.ndarray, phase: int) -> int | None:
    """Com 2 tempos por compasso detectados, decide se e 4/4 olhando onde a harmonia muda.

    Em 4/4 o acorde costuma trocar na barra de compasso: a novidade harmonica entre
    blocos de 2 tempos fica concentrada numa das paridades. Devolve a nova fase para
    4/4, ou None se continua 2/4.
    """
    if len(beats) < 16:
        return None
    pos = np.interp([n.start for n in notes], beats, np.arange(len(beats)))
    n_seg = (len(beats) - phase) // 2
    chroma = np.zeros((n_seg, 12))
    bass = np.full(n_seg, 999)
    for n, p in zip(notes, pos):
        k = int((p - phase + 0.1) // 2)
        if 0 <= k < n_seg:
            chroma[k, n.pitch % 12] += min(n.end - n.start, 1.5)
            bass[k] = min(bass[k], n.pitch)
    for k in range(n_seg):
        if bass[k] < 999:
            chroma[k, bass[k] % 12] += chroma[k].sum() * 0.5  # baixo pesa mais
    norm = np.linalg.norm(chroma, axis=1)
    nov = {0: [], 1: []}
    for k in range(1, n_seg):
        if norm[k] > 0 and norm[k - 1] > 0:
            nov[k % 2].append(1 - chroma[k] @ chroma[k - 1] / (norm[k] * norm[k - 1]))
    if min(len(nov[0]), len(nov[1])) < 4:
        return None
    even, odd = float(np.mean(nov[0])), float(np.mean(nov[1]))
    if max(even, odd) < 1.3 * min(even, odd) or max(even, odd) < 0.1:
        return None
    return phase % 4 if even > odd else (phase + 2) % 4


def scale_beats(beats: np.ndarray, factor: float) -> np.ndarray:
    if factor == 2:
        mids = (beats[:-1] + beats[1:]) / 2
        return np.sort(np.concatenate([beats, mids]))
    if factor == 0.5:
        return beats[::2]
    return beats


def build_beat_grid(beats: np.ndarray, phase: int, bpb: int, t_first: float, t_last: float) -> np.ndarray:
    """Estende a grade para cobrir todas as notas e alinha o indice 0 a um tempo forte."""
    b = list(beats)
    p0 = float(np.median(np.diff(b[:5])))
    while b[0] - t_first > 0.0625 * p0:  # nota bem antes do 1o tempo detectado (anacruse)
        b.insert(0, b[0] - p0)
        phase = (phase + 1) % bpb
    while phase % bpb:
        b.insert(0, b[0] - p0)
        phase = (phase + 1) % bpb
    while len(b) > bpb + 1 and t_first >= b[bpb] - 0.0625 * p0:  # compasso inicial vazio
        del b[:bpb]
    p1 = float(np.median(np.diff(b[-5:])))
    while b[-1] < t_last + p1:
        b.append(b[-1] + p1)
    while (len(b) - 1) % bpb:
        b.append(b[-1] + p1)
    return np.array(b)


def constant_grid(bpm: float, t_first: float, t_last: float) -> np.ndarray:
    period = 60.0 / bpm
    return np.arange(t_first, t_last + 2 * period, period)


# --------------------------------------------------------- quantizacao

# custo por nota de cada subdivisao do tempo (penaliza grades mais complexas)
SUBDIV_COST = {
    False: {1: 0.0, 2: 0.05, 4: 0.15, 3: 0.25, 8: 1.5},  # compasso simples
    True: {1: 0.0, 3: 0.05, 6: 0.15, 2: 0.4, 12: 1.5},  # compasso composto
}
TIMING_SIGMA = 0.04  # desvio tipico de um interprete, em tempos
TIMING_CAP = 0.015  # erro quadratico maximo por nota (outliers nao dominam)


def _family(s: int, compound: bool):
    if s == 1:
        return None
    if compound:
        return "b" if s == 2 else "t"
    return "t" if s % 3 == 0 else "b"


class Quantizer:
    """Escolhe uma subdivisao por tempo (Viterbi) e arredonda posicoes para ela."""

    WIN = 1 / 16  # notas ate 1/16 de tempo adiantadas pertencem ao tempo seguinte

    def __init__(self, onset_pos, n_beats: int, compound: bool, switch_cost=1.0, collide_cost=2.0):
        self.compound = compound
        subs = list(SUBDIV_COST[compound])
        self.windows: dict[int, list[float]] = defaultdict(list)
        for p in onset_pos:
            j = math.floor(p + self.WIN)
            self.windows[j].append(p - j)
        n = max(n_beats, max(self.windows, default=0) + 1) + 2
        emis = np.zeros((n, len(subs)))
        for j, fr in self.windows.items():
            if not 0 <= j < n:
                continue
            fr.sort()
            for k, s in enumerate(subs):
                c, prev = 0.0, None
                for f in fr:
                    q = round(f * s) / s
                    c += min((f - q) ** 2, TIMING_CAP) / (2 * TIMING_SIGMA ** 2) + SUBDIV_COST[compound][s]
                    if prev is not None and q == prev:  # duas notas distintas no mesmo ponto
                        c += collide_cost
                    prev = q
                emis[j, k] = c
        # estados (subdivisao, contexto binario/ternario): um tempo so com notas na
        # cabeca nao "esquece" o contexto, entao uma tercina isolada paga a troca
        states = [(1, "b"), (1, "t")] + [(s, _family(s, compound)) for s in subs if s != 1]
        col = [subs.index(s) for s, _ in states]
        trans = np.array([[switch_cost if fa != fb else 0.0 for _, fb in states] for _, fa in states])
        default = "t" if compound else "b"
        dp = emis[0, col] + np.array([0.0 if f == default else switch_cost for _, f in states])
        back = np.zeros((n, len(states)), dtype=int)
        for j in range(1, n):
            tot = dp[:, None] + trans
            back[j] = tot.argmin(axis=0)
            dp = tot.min(axis=0) + emis[j, col]
        path = [int(dp.argmin())]
        for j in range(n - 1, 0, -1):
            path.append(int(back[j][path[-1]]))
        path.reverse()
        self.grid = [states[k][0] for k in path]
        self.subs, self.emis = subs, emis

    def cost(self, j: int, s: int) -> float:
        return float(self.emis[j, self.subs.index(s)])

    def q(self, p: float) -> Fraction:
        j = min(max(math.floor(p + self.WIN), 0), len(self.grid) - 1)
        s = self.grid[j]
        return Fraction(j) + Fraction(round((p - j) * s), s)

    def unit(self, qpos: Fraction) -> Fraction:
        j = min(max(math.floor(qpos), 0), len(self.grid) - 1)
        return Fraction(1, self.grid[j])

    def family_counts(self) -> tuple[int, int]:
        """(tempos ternarios, tempos binarios) entre os que tem 2+ ataques."""
        tern = bin_ = 0
        for j, fr in self.windows.items():
            if len(fr) < 2 or not 0 <= j < len(self.grid):
                continue
            fam = _family(self.grid[j], self.compound)
            tern += fam == "t"
            bin_ += fam == "b"
        return tern, bin_


REFINE_REGULARITY = 20.0  # custo de deixar dois tempos vizinhos com duracoes diferentes


def _beat_fit_cost(fr: np.ndarray, compound: bool, collide_cost: float = 2.0, switch_cost: float = 2.0) -> float:
    """Custo do melhor encaixe das fracoes `fr` (ja ordenadas) numa subdivisao do tempo.

    A familia fora do padrao do compasso (tercina no simples, duina no composto)
    paga a ida e a volta da troca de grade, como no Viterbi.
    """
    if len(fr) == 0:
        return 0.0
    default = "t" if compound else "b"
    best = math.inf
    for s, lam in SUBDIV_COST[compound].items():
        q = np.round(fr * s) / s
        c = float(np.minimum((fr - q) ** 2, TIMING_CAP).sum()) / (2 * TIMING_SIGMA ** 2) + lam * len(fr)
        if _family(s, compound) not in (None, default):
            c += switch_cost
        c += collide_cost * int(np.sum(np.diff(q) == 0))
        best = min(best, c)
    return best


def refine_grid(grid: np.ndarray, onset_times, compound: bool, max_shift: float = 0.3, steps: int = 13) -> np.ndarray:
    """Corrige tempos mal posicionados pelo detector (Viterbi sobre o deslocamento de cada tempo).

    Cada tempo pode andar ate `max_shift` do periodo local. O custo de cada intervalo
    e o encaixe dos seus ataques numa grade simples mais a irregularidade frente ao
    andamento local; como a busca e global, um trecho inteiro defasado de uma
    semicolcheia (comum no inicio, antes do detector "travar") e corrigido em bloco.
    """
    g = np.array(grid, dtype=float)
    n = len(g)
    if n < 4:
        return g
    on = np.sort(np.asarray(onset_times, dtype=float))
    d = np.diff(g)
    period = np.array([np.median(d[max(0, i - 4): i + 5]) for i in range(n - 1)])
    beat_period = np.concatenate([period[:1], (period[:-1] + period[1:]) / 2, period[-1:]])
    cand = np.linspace(-max_shift, max_shift, steps)
    pos = g[:, None] + cand[None, :] * beat_period[:, None]  # posicao candidata de cada tempo
    node = 0.1 * np.abs(cand)
    dp = node.copy()
    back = np.zeros((n, steps), dtype=int)
    for j in range(n - 1):
        seg = on[np.searchsorted(on, pos[j].min() - period[j] / 4): np.searchsorted(on, pos[j + 1].max())]
        trans = np.full((steps, steps), math.inf)
        for ia in range(steps):
            x0 = pos[j, ia]
            for ib in range(steps):
                dur = pos[j + 1, ib] - x0
                if dur < 0.4 * period[j]:
                    continue
                w = dur / 16
                fr = (seg[(seg >= x0 - w) & (seg < x0 + dur - w)] - x0) / dur
                trans[ia, ib] = (_beat_fit_cost(fr, compound)
                                 + REFINE_REGULARITY * ((dur - period[j]) / period[j]) ** 2)
        tot = dp[:, None] + trans
        back[j + 1] = tot.argmin(axis=0)
        dp = tot.min(axis=0) + node
    k = int(dp.argmin())
    out = np.empty(n)
    for j in range(n - 1, -1, -1):
        out[j] = pos[j, k]
        k = back[j, k]
    return out


def cluster_onsets(notes: list[Note], pos, window: float = 0.04) -> list[float]:
    """Notas tocadas quase juntas (acorde, arpejo rapido) recebem a mesma posicao."""
    order = sorted(range(len(notes)), key=lambda i: notes[i].start)
    rep = [0.0] * len(notes)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and notes[order[j + 1]].start - notes[order[i]].start <= window:
            j += 1
        members = order[i: j + 1]
        p = float(np.median([pos(notes[m].start) for m in members]))
        for m in members:
            rep[m] = p
        i = j + 1
    return rep


POLY_MARGIN = 2.0  # evidencia minima para uma mao seguir grade propria (polirritmia)


def hand_quantizers(notes: list[Note], onset_pos: list[float], n_beats: int, compound: bool):
    """Grade comum as duas maos (uma confirma a outra), com excecao para polirritmia.

    Uma mao so ganha grade propria num tempo quando tem 3+ ataques nele e eles
    apontam claramente para a outra familia (tercinas contra colcheias).
    Devolve (quantizador por mao, quantizador global).
    """
    glob = Quantizer(onset_pos, n_beats, compound)
    out = {}
    for h in (RIGHT, LEFT):
        own = Quantizer([p for n, p in zip(notes, onset_pos) if n.hand == h], n_beats, compound)
        grid = list(glob.grid)
        for j, fr in own.windows.items():
            if len(fr) < 3 or not 0 <= j < min(len(grid), len(own.grid)):
                continue
            fo, fg = _family(own.grid[j], compound), _family(grid[j], compound)
            if fo and fg and fo != fg and own.cost(j, grid[j]) - own.cost(j, own.grid[j]) > POLY_MARGIN:
                grid[j] = own.grid[j]
        own.grid = grid
        out[h] = own
    return out, glob


def quantize_notes(notes: list[Note], qzs: dict, onset_pos: list[float], pos) -> None:
    for n, p in zip(notes, onset_pos):
        qz = qzs[n.hand]
        n.q_start = qz.q(p)
        n.q_end = qz.q(pos(n.end))
        if n.q_end <= n.q_start:
            n.q_end = n.q_start + qz.unit(n.q_start)


def resolve_overlaps(notes: list[Note]) -> list[Note]:
    """Remove duplicatas e corta notas iguais que se sobrepoem na mesma mao."""
    by_key: dict = {}
    for n in notes:
        k = (n.hand, n.pitch, n.q_start)
        if k in by_key:
            o = by_key[k]
            o.q_end = max(o.q_end, n.q_end)
            o.end = max(o.end, n.end)
            o.velocity = max(o.velocity, n.velocity)
        else:
            by_key[k] = n
    out = sorted(by_key.values(), key=lambda n: (n.q_start, n.pitch))
    last: dict = {}
    for n in out:
        k = (n.hand, n.pitch)
        if k in last and last[k].q_end > n.q_start:
            last[k].q_end = n.q_start
        last[k] = n
    return out


# --------------------------------------------------------------- tonalidade

KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
MAJOR_KEYS = ["C", "D-", "D", "E-", "E", "F", "F#", "G", "A-", "A", "B-", "B"]
MINOR_KEYS = ["c", "c#", "d", "e-", "e", "f", "f#", "g", "g#", "a", "b-", "b"]


def detect_key(notes: list[Note]) -> str:
    """Krumhansl-Schmuckler sobre as classes de altura ponderadas por duracao."""
    hist = np.zeros(12)
    for n in notes:
        hist[n.pitch % 12] += min(n.end - n.start, 2.0) * (0.3 + n.velocity / 127)
    if hist.sum() == 0:
        return "C"
    best, name = -2.0, "C"
    for tonic in range(12):
        for prof, names in ((KS_MAJOR, MAJOR_KEYS), (KS_MINOR, MINOR_KEYS)):
            r = np.corrcoef(hist, np.roll(prof, tonic))[0, 1]
            if r > best:
                best, name = r, names[tonic]
    return name


def mido_key(m21_key: str) -> str:
    name = m21_key[0].upper() + m21_key[1:].replace("-", "b")
    return name + ("m" if m21_key[0].islower() else "")


def make_speller(m21_key: str):
    """Grafia das notas coerente com a armadura (Bb em Fa maior, C# em Re menor...)."""
    from music21 import key, pitch

    k = key.Key(m21_key)
    names: dict[int, str] = {}
    for p in k.getScale(k.mode).getPitches()[:7]:
        names[p.pitchClass] = p.name
    if k.mode == "minor":
        for iv in ("M6", "M7"):  # 6a e 7a elevadas da menor melodica/harmonica
            p = k.tonic.transpose(iv)
            names.setdefault(p.pitchClass, p.name)
    for pc in range(12):
        if pc not in names:
            p = pitch.Pitch(midi=60 + pc)
            if k.sharps < 0 and p.accidental is not None and p.accidental.alter > 0:
                p = p.getEnharmonic()
            names[pc] = p.name
    cache: dict[int, object] = {}

    def spell(midi: int):
        if midi not in cache:
            p = pitch.Pitch(names[midi % 12])
            base = midi // 12 - 1
            for octave in (base, base - 1, base + 1):
                p.octave = octave
                if round(p.ps) == midi:
                    break
            cache[midi] = p
        return pitch.Pitch(cache[midi].nameWithOctave)

    return spell


# ------------------------------------------------------ vozes e partitura


@dataclass
class Elem:
    start: Fraction
    end: Fraction
    pitches: list
    release: float = 0.0  # soltura real mais tardia (s), para checar o pedal
    tie: str | None = None

    @property
    def rest(self) -> bool:
        return not self.pitches


def _mean(ps) -> float:
    return sum(ps) / len(ps)


HOLD_BEATS = Fraction(1)  # nota que passa 1+ tempo do ataque seguinte vira voz sustentada


def build_voices(notes: list[Note], max_voices: int, pedal_release, legato: bool) -> list[list[Elem]]:
    """Voz principal: um acorde por ataque, ate o ataque seguinte da mao.

    Notas seguradas por cima de outros ataques (baixo sustentado, melodia longa)
    vao para vozes extras, ate `max_voices`; o resto da voz principal fica limpo,
    sem a colcha de ligaduras e pausas que a duracao literal de cada tecla geraria.
    """
    if not notes:
        return []
    onsets = sorted({n.q_start for n in notes})
    next_onset = {t: onsets[i + 1] if i + 1 < len(onsets) else None for i, t in enumerate(onsets)}
    main, held = [], []
    for n in notes:
        nxt = next_onset[n.q_start]
        if max_voices > 1 and nxt is not None and n.q_end - nxt >= HOLD_BEATS:
            held.append(n)
        else:
            main.append(n)

    by_on: dict[Fraction, list[Note]] = defaultdict(list)
    for n in main:
        by_on[n.q_start].append(n)
    starts = sorted(by_on)
    first: list[Elem] = []
    for i, t in enumerate(starts):
        grp = by_on[t]
        end = max(n.q_end for n in grp)
        if i + 1 < len(starts):
            end = min(end, starts[i + 1])
        first.append(Elem(t, end, sorted({n.pitch for n in grp}), max(n.end for n in grp)))

    extra: list[list[Elem]] = []
    for n in sorted(held, key=lambda n: (n.q_start, -n.pitch)):
        e = Elem(n.q_start, n.q_end, [n.pitch], n.end)
        for v in extra:
            last = v[-1]
            if last.start == e.start:  # mesmo ataque: acorde com a duracao mais longa
                last.pitches = sorted(set(last.pitches) | {n.pitch})
                last.end = max(last.end, e.end)
                last.release = max(last.release, e.release)
                break
            if last.end <= e.start:
                v.append(e)
                break
        else:
            if len(extra) < max_voices - 1:
                extra.append([e])
            else:  # sem voz livre: corta a nota sustentada que termina primeiro
                v = min(extra, key=lambda v: v[-1].end)
                v[-1].end = e.start
                v.append(e)
    voices = ([first] if first else []) + extra

    if legato:  # preenche pausas curtas e as cobertas pelo pedal
        for voice in voices:
            for a, b in zip(voice, voice[1:]):
                gap = b.start - a.end
                if gap <= 0:
                    continue
                rel = pedal_release(a.release)
                if rel is not None:
                    a.end = min(b.start, max(a.end, rel))
                elif gap <= Fraction(1, 4):
                    a.end = b.start
            last = voice[-1]
            rel = pedal_release(last.release)
            if rel is not None:
                last.end = max(last.end, min(rel, last.end + 4))

    def weight(v):
        tot = sum(float(e.end - e.start) for e in v)
        return sum(_mean(e.pitches) * float(e.end - e.start) for e in v) / tot if tot else 0

    voices.sort(key=weight, reverse=True)  # voz 1 = a mais aguda (hastes para cima)
    return voices


def _nice_whole(s: Fraction, d: Fraction, compound: bool) -> bool:
    """A figura pode ser escrita inteira, sem ligar no meio do tempo?"""
    fs = s - math.floor(s)
    if compound:
        return fs == 0 and d.denominator == 1
    if fs == 0:
        return d in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), 1, Fraction(3, 2), 2, 3, 4)
    if fs == Fraction(1, 2):
        return d in (1, Fraction(3, 2))
    return False


def split_elem(e: Elem, bpb: int, compound: bool) -> list[Elem]:
    cuts = {Fraction(k * bpb) for k in range(math.floor(e.start / bpb) + 1, math.ceil(e.end / bpb))}
    bounds = sorted({e.start, e.end} | cuts)
    segs = []
    for s, t in zip(bounds, bounds[1:]):
        if e.rest:
            inner = {Fraction(k) for k in range(math.ceil(s), math.floor(t) + 1) if s < k < t}
            pts = sorted({s, t} | inner)
            segs.extend(zip(pts, pts[1:]))
        elif _nice_whole(s, t - s, compound):
            segs.append((s, t))
        else:
            inner = set()
            if s.denominator != 1 and math.ceil(s) < t:
                inner.add(Fraction(math.ceil(s)))
            if t.denominator != 1 and math.floor(t) > s:
                inner.add(Fraction(math.floor(t)))
            pts = sorted({s, t} | inner)
            segs.extend(zip(pts, pts[1:]))
    if e.rest:
        segs = _merge_rests(segs, bpb, compound)
    out = []
    for i, (s, t) in enumerate(segs):
        tie = None
        if not e.rest and len(segs) > 1:
            tie = "start" if i == 0 else ("stop" if i == len(segs) - 1 else "continue")
        out.append(Elem(s, t, e.pitches, e.release, tie))
    return out


def _merge_rests(segs, bpb: int, compound: bool):
    """Pausas de um tempo viram pausa de meio compasso / compasso inteiro quando cabem."""
    out = []
    for s, t in segs:
        if out:
            ps, pt = out[-1]
            bar = math.floor(ps / bpb) * bpb
            same_bar = math.floor(s / bpb) * bpb == bar
            if same_bar and pt == s and ps.denominator == 1 and t.denominator == 1:
                if (t - bar == bpb and ps == bar) or (bpb == 4 and not compound and (ps - bar) % 2 == 0 and t - ps == 2):
                    out[-1] = (ps, t)
                    continue
        out.append((s, t))
    return out


def voice_to_measures(voice: list[Elem], n_measures: int, bpb: int, compound: bool) -> list[list[Elem]]:
    total = Fraction(n_measures * bpb)
    timeline: list[Elem] = []
    cur = Fraction(0)
    for e in voice:
        if e.start >= total:
            break
        if e.start > cur:
            timeline.append(Elem(cur, e.start, []))
        e.end = min(e.end, total)
        timeline.append(e)
        cur = e.end
    if cur < total:
        timeline.append(Elem(cur, total, []))
    measures: list[list[Elem]] = [[] for _ in range(n_measures)]
    for e in timeline:
        for piece in split_elem(e, bpb, compound):
            measures[min(int(piece.start // bpb), n_measures - 1)].append(piece)
    return measures


def choose_clefs(measure_pitches: list[list[int]], start: str) -> list[str]:
    clefs, cur = [], start
    for ps in measure_pitches:
        if ps:
            med = float(np.median(ps))
            if cur == "bass" and min(ps) >= 55 and med >= 62:
                cur = "treble"
            elif cur == "treble" and max(ps) <= 64 and med <= 53:
                cur = "bass"
        clefs.append(cur)
    return clefs


def build_score(title, subtitle, meter: Meter, key_name, bpm, hand_voices, n_measures, pedal_spans, pedal_marks):
    from music21 import chord, clef, duration, expressions, instrument, key, layout, metadata
    from music21 import meter as m21meter
    from music21 import note, stream, tempo, tie

    spell = make_speller(key_name)
    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = title
    score.metadata.composer = subtitle
    staffs = []
    for h in (RIGHT, LEFT):
        ps = stream.PartStaff(id=("MD", "ME")[h])
        ps.insert(0, instrument.Piano())
        per_voice = [voice_to_measures(v, n_measures, meter.bpb, meter.compound) for v in hand_voices[h]]
        if not per_voice:
            per_voice = [voice_to_measures([], n_measures, meter.bpb, meter.compound)]
        m_pitches = [[p for v in per_voice for e in v[i] for p in e.pitches] for i in range(n_measures)]
        clefs = choose_clefs(m_pitches, "treble" if h == RIGHT else "bass")
        placed: list[tuple[Fraction, object]] = []
        for i in range(n_measures):
            m = stream.Measure(number=i + 1)
            if i == 0 or clefs[i] != clefs[i - 1]:
                m.insert(0, clef.TrebleClef() if clefs[i] == "treble" else clef.BassClef())
            if i == 0:
                m.insert(0, key.Key(key_name))
                m.insert(0, m21meter.TimeSignature(str(meter)))
                if h == RIGHT:
                    m.insert(0, tempo.MetronomeMark(number=bpm, referent=duration.Duration(meter.beat_ql)))
            active = [v for v in per_voice if any(not e.rest for e in v[i])] or per_voice[:1]
            containers = [m] if len(active) == 1 else [stream.Voice(id=str(k + 1)) for k in range(len(active))]
            bar_start = Fraction(i * meter.bpb)
            for vi, (v, cont) in enumerate(zip(active, containers)):
                pieces = v[i]
                full_rest = len(pieces) == 1 and pieces[0].rest
                for e in pieces:
                    ql = (e.end - e.start) * meter.beat_ql
                    if e.rest:
                        obj = note.Rest(quarterLength=ql)
                        if full_rest and vi == 0:
                            obj.fullMeasure = True
                        if vi > 0:
                            obj.style.hideObjectOnPrint = True
                    elif len(e.pitches) == 1:
                        obj = note.Note(spell(e.pitches[0]), quarterLength=ql)
                    else:
                        obj = chord.Chord([spell(p) for p in e.pitches], quarterLength=ql)
                    if e.tie:
                        obj.tie = tie.Tie(e.tie)
                    cont.insert((e.start - bar_start) * meter.beat_ql, obj)
                    if not e.rest:
                        placed.append((e.start, obj))
                if cont is not m:
                    m.insert(0, cont)
            ps.append(m)
        if pedal_marks and h == LEFT:
            placed.sort(key=lambda x: x[0])
            starts = [p[0] for p in placed]
            for a, b in pedal_spans:
                lo = np.searchsorted(starts, a, side="left")
                hi = np.searchsorted(starts, b, side="left")
                if hi - lo >= 1:
                    objs = [placed[k][1] for k in range(lo, hi)]
                    pm = expressions.PedalMark(objs[0], objs[-1]) if len(objs) > 1 else expressions.PedalMark(objs[0])
                    pm.pedalForm = expressions.PedalForm.Symbol
                    ps.insert(0, pm)
        staffs.append(ps)
    for ps in staffs:
        score.insert(0, ps)
    score.insert(0, layout.StaffGroup(staffs, name="Piano", abbreviation="Pno.", symbol="brace"))
    return score


# ------------------------------------------------------------------- MIDI


def write_midi(path: Path, title: str, meter: Meter, key_name: str, beat_times: np.ndarray,
               notes: list[tuple[int, int, int, int, int]], pedal: list[tuple[int, int]]) -> None:
    """MIDI tipo 1: faixa de andamento + 'Right Hand' (canal 1) + 'Left Hand' (canal 2)."""
    import mido

    mid = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    beat_ticks = int(meter.beat_ql * PPQ)

    midi_title = title[:60].encode("latin-1", "replace").decode("latin-1")  # texto MIDI e latin-1
    conductor = [
        (0, 0, mido.MetaMessage("track_name", name=midi_title)),
        (0, 0, mido.MetaMessage("time_signature", numerator=meter.num, denominator=meter.den,
                                clocks_per_click=36 if meter.compound else 24, notated_32nd_notes_per_beat=8)),
        (0, 0, mido.MetaMessage("key_signature", key=mido_key(key_name))),
    ]
    last_tempo = None
    for i in range(len(beat_times) - 1):
        us_per_quarter = int(round((beat_times[i + 1] - beat_times[i]) / float(meter.beat_ql) * 1e6))
        us_per_quarter = min(max(us_per_quarter, 100_000), 4_000_000)
        if us_per_quarter != last_tempo:
            conductor.append((i * beat_ticks, 1, mido.MetaMessage("set_tempo", tempo=us_per_quarter)))
            last_tempo = us_per_quarter
    mid.tracks.append(_to_track(conductor))

    for hand in (RIGHT, LEFT):
        ch = hand
        ev = [
            (0, 0, mido.MetaMessage("track_name", name=HAND_NAMES[hand])),
            (0, 0, mido.Message("program_change", program=0, channel=ch)),
        ]
        hand_notes = sorted((n for n in notes if n[4] == hand), key=lambda n: (n[2], n[0]))
        prev: dict[int, list] = {}
        fixed = []
        for on, off, pitch, vel, _ in hand_notes:  # mesma tecla: corta a anterior
            if pitch in prev and prev[pitch][1] > on:
                prev[pitch][1] = on
            item = [on, max(off, on + 1), pitch, vel]
            fixed.append(item)
            prev[pitch] = item
        for on, off, pitch, vel in fixed:
            if off <= on:
                continue
            ev.append((on, 2, mido.Message("note_on", note=pitch, velocity=max(1, min(127, vel)), channel=ch)))
            ev.append((off, 0, mido.Message("note_off", note=pitch, velocity=0, channel=ch)))
        for down, up in pedal:
            ev.append((down, 1, mido.Message("control_change", control=64, value=127, channel=ch)))
            ev.append((up, 1, mido.Message("control_change", control=64, value=0, channel=ch)))
        mid.tracks.append(_to_track(ev))
    mid.save(str(path))


def _to_track(events):
    import mido

    track = mido.MidiTrack()
    now = 0
    for tick, _, msg in sorted(events, key=lambda e: (e[0], e[1])):
        track.append(msg.copy(time=tick - now))
        now = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


# ------------------------------------------------------------ renderizacao


def find_musescore() -> str | None:
    for name in ("mscore", "MuseScore4", "MuseScore3", "musescore", "mscore4portable"):
        exe = shutil.which(name)
        if exe:
            return exe
    candidates = [
        r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
        r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
        "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
        "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
    ]
    return next((c for c in candidates if Path(c).exists()), None)


def find_browser() -> str | None:
    """Edge/Chrome em modo headless imprimem o HTML do Verovio em PDF."""
    for name in ("msedge", "chrome", "google-chrome", "chromium", "chromium-browser"):
        exe = shutil.which(name)
        if exe:
            return exe
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ]
    return next((c for c in candidates if Path(c).exists()), None)


def render_score(musicxml: Path, pdf: Path, html: Path) -> list[str]:
    """PDF pelo MuseScore se instalado; senao Verovio -> HTML -> PDF pelo Edge/Chrome."""
    ms = find_musescore()
    if ms:
        try:
            subprocess.run([ms, "-o", str(pdf), str(musicxml)], check=True, timeout=600,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if pdf.exists():
                return [f"{pdf} (MuseScore)"]
        except Exception as exc:
            log(f"  MuseScore falhou ({exc}); usando Verovio.")
    try:
        import verovio
    except ImportError:
        return []
    verovio.enableLog(verovio.LOG_ERROR)
    tk = verovio.toolkit()
    tk.setOptions({
        "pageWidth": 2100, "pageHeight": 2970, "scale": 40, "adjustPageHeight": False,
        "pageMarginLeft": 100, "pageMarginRight": 100, "pageMarginTop": 80, "pageMarginBottom": 80,
        "footer": "none", "breaks": "auto",
    })
    if not tk.loadFile(str(musicxml)):
        log("  Verovio nao conseguiu ler o MusicXML.")
        return []
    pages = []
    with tempfile.TemporaryDirectory(prefix="piano2sheet_svg_") as tmp:
        for p in range(1, tk.getPageCount() + 1):  # via arquivo: a string do binding perde o UTF-8 no Windows
            svg_file = Path(tmp) / f"{p}.svg"
            tk.renderToSVGFile(str(svg_file), p)
            pages.append(svg_file.read_text(encoding="utf-8").replace("â€“", "–"))  # "- 2 -"
    body = "\n".join(f'<div class="page">{svg}</div>' for svg in pages)
    html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>"
        + musicxml.stem
        + "</title><style>body{background:#888;margin:0}"
        ".page{background:#fff;width:210mm;height:297mm;margin:8mm auto;box-shadow:0 0 6px #0006;overflow:hidden}"
        ".page svg{width:100%;height:auto;display:block}"
        "@media print{body{background:#fff}.page{margin:0;box-shadow:none;break-after:page}}"
        "@page{size:A4;margin:0}</style></head><body>" + body + "</body></html>",
        encoding="utf-8",
    )
    made = [f"{html} (Verovio)"]
    browser = find_browser()
    if browser:
        with tempfile.TemporaryDirectory(prefix="piano2sheet_browser_") as profile:
            try:
                subprocess.run([browser, "--headless", "--disable-gpu", f"--user-data-dir={profile}",
                                "--no-pdf-header-footer", "--print-to-pdf-no-header",
                                f"--print-to-pdf={pdf.resolve()}", html.resolve().as_uri()],
                               check=True, timeout=180, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as exc:
                log(f"  Impressao do PDF pelo navegador falhou ({exc}).")
        if pdf.exists():
            made.insert(0, f"{pdf} (Verovio)")
    if len(made) == 1:
        made[0] += " - abra no navegador e use Ctrl+P para salvar em PDF"
    return made


# -------------------------------------------------------------- principal


def acquire(args, out_root: Path):
    """Resolve titulo e pasta de saida; baixa/converte o audio so se precisar."""
    if is_url(args.source):
        log("Lendo informacoes do video...")
        info = youtube_info(args.source)
        title = info.get("title") or info.get("id") or "video"
        author = info.get("uploader") or info.get("channel") or ""
    else:
        src = Path(args.source)
        if not src.exists():
            raise SystemExit(f"Arquivo nao encontrado: {src}")
        title, author = src.stem, ""
    name = safe_name(title)
    outdir = out_root / name
    outdir.mkdir(parents=True, exist_ok=True)
    return title, author, name, outdir


def run_neural(args, outdir: Path):
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    device = pick_device(args.device)
    start, end = parse_time(args.start), parse_time(args.end)
    with tempfile.TemporaryDirectory(prefix="piano2sheet_") as tmp:
        tmp = Path(tmp)
        if is_url(args.source):
            log("Baixando audio (yt-dlp)...")
            src = download_audio(args.source, tmp)
        else:
            src = Path(args.source)
        wav = outdir / "audio.wav" if args.keep_audio else tmp / "audio.wav"
        to_wav(src, wav, start, end)
        audio, sr = load_wav(wav)
        log(f"Audio: {len(audio) / sr:.1f} s. Transcrevendo com Transkun ({device})...")
        notes, pedals = transcribe_audio(audio, sr, device)
        log(f"  {len(notes)} notas, {len(pedals)} pedaladas. Detectando tempos (Beat This!)...")
        try:
            beats, downbeats = track_beats(audio, sr, device)
        except Exception as exc:
            log(f"  Deteccao de tempos falhou ({exc}); usando andamento constante.")
            beats, downbeats = np.array([]), np.array([])
    return notes, pedals, beats, downbeats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Transcreve piano de um video do YouTube (ou arquivo de audio) para partitura e MIDI Synthesia.")
    ap.add_argument("source", help="link do YouTube ou arquivo de audio/video local")
    ap.add_argument("-o", "--output", default="saida", help="pasta de saida (padrao: ./saida)")
    ap.add_argument("--start", help="inicio do trecho (s ou mm:ss)")
    ap.add_argument("--end", help="fim do trecho (s ou mm:ss)")
    ap.add_argument("--device", default="auto", help="auto, cuda ou cpu")
    ap.add_argument("--time-signature", help="forca a formula de compasso, ex.: 4/4, 3/4, 6/8")
    ap.add_argument("--bpm", type=float, help="ignora a deteccao e usa andamento constante (gravacao com metronomo)")
    ap.add_argument("--beat-scale", type=float, choices=[0.5, 1, 2], default=None,
                    help="2 = dobra os tempos detectados, 0.5 = usa metade (corrige andamento dobrado/pela metade)")
    ap.add_argument("--split-point", type=int, help="divide as maos por altura MIDI fixa (ex.: 60) em vez da busca")
    ap.add_argument("--voices", type=int, default=2, help="maximo de vozes por pauta (padrao 2)")
    ap.add_argument("--min-velocity", type=int, default=8, help="descarta notas mais fracas que isso")
    ap.add_argument("--no-legato", action="store_true", help="nao prolonga notas cobertas pelo pedal na partitura")
    ap.add_argument("--no-pedal-marks", action="store_true", help="nao escreve marcas de pedal na partitura")
    ap.add_argument("--no-pdf", action="store_true", help="nao renderiza PDF/HTML")
    ap.add_argument("--keep-audio", action="store_true", help="guarda o WAV na pasta de saida")
    ap.add_argument("--force", action="store_true", help="refaz a transcricao mesmo com cache")
    args = ap.parse_args(argv)

    title, author, name, outdir = acquire(args, Path(args.output))
    cache = outdir / "transcricao.json"
    key = {"source": args.source, "start": args.start, "end": args.end, "version": CACHE_VERSION}
    data = None
    if cache.exists() and not args.force:
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("key") != key:
            data = None
        else:
            log("Usando transcricao em cache (--force para refazer).")
    if data is None:
        notes, pedals, beats, downbeats = run_neural(args, outdir)
        data = {
            "key": key, "title": title, "author": author,
            "notes": [[n.start, n.end, n.pitch, n.velocity] for n in notes],
            "pedals": pedals, "beats": beats.tolist(), "downbeats": downbeats.tolist(),
        }
        cache.write_text(json.dumps(data), encoding="utf-8")

    result = arrange(data, args)
    write_outputs(result, data, args, outdir, name)


def arrange(data: dict, args) -> dict:
    notes = [Note(s, e, p, v) for s, e, p, v in data["notes"] if v >= args.min_velocity and 21 <= p <= 108]
    for n in notes:
        n.end = max(n.end, n.start + 0.02)
    if not notes:
        raise SystemExit("Nenhuma nota detectada.")
    pedals = [tuple(p) for p in data["pedals"]]
    t_first = min(n.start for n in notes)
    t_last = max(n.end for n in notes)

    log("Separando as maos...")
    if args.split_point:
        split_hands_fixed(notes, args.split_point)
    else:
        split_hands(notes)

    user_meter = Meter.parse(args.time_signature) if args.time_signature else None
    beats = clean_beats(np.array(data["beats"]))
    downbeats = np.array(data["downbeats"])
    if args.bpm or len(beats) < 8:
        bpm = args.bpm or 100.0
        beats = constant_grid(bpm, t_first, t_last)
        meter = user_meter or Meter.simple(4)
        phase = 0
    else:
        scale = args.beat_scale
        if scale is None:  # tempo detectado lento demais para as figuras tocadas?
            pos0 = lambda t: np.interp(t, beats, np.arange(len(beats)))  # noqa: E731
            iois = np.diff(np.unique(np.round([pos0(n.start) for n in notes], 3)))
            iois = iois[iois > 0.02]
            bpm0 = 60.0 / float(np.median(np.diff(beats)))
            if len(iois) and bpm0 < 72 and np.mean(iois < 0.2) > 0.3:
                scale = 2
                log(f"  Andamento detectado ({bpm0:.0f} bpm) parece metade do real; dobrando os tempos.")
        if scale and scale != 1:
            beats = scale_beats(beats, scale)
        if user_meter:
            meter = user_meter
            _, phase = estimate_meter(beats, downbeats, user_meter.bpb)
        else:
            bpb, phase = estimate_meter(beats, downbeats)
            if bpb == 2:
                four = harmonic_bar_check(notes, beats, phase)
                if four is not None:
                    bpb, phase = 4, four
                    log("  Harmonia muda a cada 4 tempos: usando 4/4.")
            meter = Meter.simple(bpb)
    grid = build_beat_grid(beats, phase, meter.bpb, t_first, t_last)
    if not args.bpm:
        starts = sorted(n.start for n in notes)
        onset_times = [starts[0]]
        for t in starts[1:]:
            if t - onset_times[-1] > 0.04:
                onset_times.append(t)
        grid = refine_grid(grid, onset_times, meter.compound)
        bpb = meter.bpb
        while len(grid) > 2 * bpb + 1 and t_first >= grid[bpb] - 0.0625 * (grid[bpb] - grid[bpb - 1]):
            grid = grid[bpb:]  # o refinamento pode ter esvaziado o 1o compasso
    pos = lambda t: float(np.interp(t, grid, np.arange(len(grid))))  # noqa: E731

    log("Quantizando o ritmo...")
    onset_pos = cluster_onsets(notes, pos)
    n_beats = len(grid) - 1
    qzs, glob = hand_quantizers(notes, onset_pos, n_beats, meter.compound)
    tern, bin_ = glob.family_counts()
    if user_meter is None and not meter.compound and tern > 1.5 * bin_ and tern >= 4:
        meter = meter.to_compound()
        log(f"  Ritmo predominantemente ternario: usando {meter}.")
        qzs, glob = hand_quantizers(notes, onset_pos, n_beats, True)
    quantize_notes(notes, qzs, onset_pos, pos)
    qz = qzs[LEFT]  # pedal fica sob a pauta da mao esquerda
    notes = resolve_overlaps(notes)

    key_name = detect_key(notes)
    max_end = max(n.q_end for n in notes)
    n_measures = max(1, math.ceil(max_end / meter.bpb))
    sec_per_beat = float(np.median(np.diff(grid)))
    bpm = int(round(60.0 / sec_per_beat))

    def pedal_release(t: float, qz: Quantizer):
        for a, b in pedals:
            if a - 0.03 <= t <= b + 0.03:
                return qz.q(pos(b))
        return None

    pedal_q = []
    for a, b in pedals:
        qa, qb = qz.q(pos(a)), qz.q(pos(b))
        if qb > qa:
            pedal_q.append((qa, qb))

    hand_voices = []
    for h in (RIGHT, LEFT):
        hn = [n for n in notes if n.hand == h]
        release = lambda t, qz=qzs[h]: pedal_release(t, qz)  # noqa: E731
        hand_voices.append(build_voices(hn, max(1, args.voices), release, not args.no_legato))

    return {
        "notes": notes, "pedals": pedals, "pedal_q": pedal_q, "grid": grid, "pos": pos, "qz": qz,
        "meter": meter, "key": key_name, "bpm": bpm, "n_measures": n_measures, "hand_voices": hand_voices,
    }


def write_outputs(r: dict, data: dict, args, outdir: Path, name: str) -> None:
    meter: Meter = r["meter"]
    grid, pos, qz = r["grid"], r["pos"], r["qz"]
    title = data.get("title") or name
    author = data.get("author") or ""
    beat_ticks = meter.beat_ql * PPQ
    n_beats_used = r["n_measures"] * meter.bpb
    tempo_grid = grid[: n_beats_used + 1] if len(grid) > n_beats_used else grid

    def tick_q(q: Fraction) -> int:
        return int(round(q * beat_ticks))

    def tick_t(t: float) -> int:
        return int(round(pos(t) * float(beat_ticks)))

    # MIDI Synthesia: notas na grade da partitura, mapa de andamento do interprete
    qnotes = [(tick_q(n.q_start), tick_q(n.q_end), n.pitch, n.velocity, n.hand) for n in r["notes"]]
    qpedal = []
    for a, b in r["pedals"]:
        up = tick_q(qz.q(pos(b))) - 2
        down = max(tick_t(a), qpedal[-1][1] + 10 if qpedal else 0)
        if up > down + 10:
            qpedal.append((down, up))
    synth_path = outdir / f"{name}.mid"
    write_midi(synth_path, title, meter, r["key"], tempo_grid, qnotes, qpedal)

    # MIDI de performance: tempo exato do audio, mesmas maos e compassos
    pnotes = [(tick_t(n.start), tick_t(n.end), n.pitch, n.velocity, n.hand) for n in r["notes"]]
    ppedal = [(tick_t(a), tick_t(b)) for a, b in r["pedals"] if tick_t(b) > tick_t(a)]
    perf_path = outdir / f"{name} (performance).mid"
    write_midi(perf_path, title, meter, r["key"], tempo_grid, pnotes, ppedal)

    log("Montando a partitura...")
    subtitle = f"Transcricao automatica - {author}" if author else "Transcricao automatica"
    score = build_score(title, subtitle, meter, r["key"], r["bpm"], r["hand_voices"], r["n_measures"],
                        r["pedal_q"], not args.no_pedal_marks)
    xml_path = outdir / f"{name}.musicxml"
    score.write("musicxml", fp=str(xml_path))

    made = []
    if not args.no_pdf:
        log("Renderizando a partitura...")
        made = render_score(xml_path, outdir / f"{name}.pdf", outdir / f"{name}.html")

    n_r = sum(1 for n in r["notes"] if n.hand == RIGHT)
    from music21 import key as m21key

    log("")
    log(f"Pronto: {title}")
    log(f"  Tonalidade: {m21key.Key(r['key']).name.replace('-', 'b')}   Compasso: {meter}   "
        f"Andamento: ~{r['bpm']} bpm   Compassos: {r['n_measures']}")
    log(f"  Notas: {len(r['notes'])} (mao direita {n_r}, mao esquerda {len(r['notes']) - n_r})")
    log(f"  MIDI Synthesia:   {synth_path}")
    log(f"  MIDI performance: {perf_path}")
    log(f"  MusicXML:         {xml_path}")
    for m in made:
        log(f"  Partitura:        {m}")


if __name__ == "__main__":
    main()
