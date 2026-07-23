import json
import time
from pathlib import Path
from typing import Optional
from weakref import ref

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from scipy.signal import find_peaks, savgol_filter
from scipy.special import comb

# storage
DATA_DIR = Path("twyst_data")
DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Twyst Backend", version="0.4.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_record_session: dict = {}
_compare_session: dict = {}
_current_mode: str = "single"  

UI_SIGNAL_WINDOW = 200
COMPARE_ANALYSIS_WINDOW = 240

_link_state: dict = {
    "secondary_connected": False,
    "main_connected": False,
    "state": "unknown",
    "last_update": None,
}
LINK_STALE_SECONDS = 5.0  

_battery_state: dict = {
    "voltage": None,
    "percent": None,
    "last_update": None,
}
BATTERY_STALE_SECONDS = 10.0


class LinkStatus(BaseModel):
    secondary_connected: bool
    main_connected: bool
    state: str = "unknown"
    timestamp: Optional[float] = None


# column layout of the single-band 9-dim state vector
# [roll, pitch, yaw, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
COMPARE_INDICES = [0, 1, 3, 4, 5, 6, 7, 8]   # drops col-2 (yaw, always 0)

# column layout of the dual-band combined 18-dim state vector:
# secondary band's 9 columns, then the main band's 9 columns (prefixed m_)
DUAL_COLUMNS = [
    "roll", "pitch", "yaw", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
    "m_roll", "m_pitch", "m_yaw", "m_acc_x", "m_acc_y", "m_acc_z", "m_gyro_x", "m_gyro_y", "m_gyro_z",
]
# drops col-2 (yaw) and col-11 (m_yaw) — both are always 0 on this hardware
COMPARE_INDICES_DUAL = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]

MAIN_FRAME_BUFFER_MAX = 500

_main_frame_buffer: list[dict] = []
_latest_main_frame: Optional[dict] = None


# pydantic models

class IMUFrame(BaseModel):
    acc_x: float = 0.0
    acc_y: float = 0.0
    acc_z: float = 1.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    ts: Optional[int] = None
    host_timestamp: Optional[float] = None
    batt_v: Optional[float] = None
    batt_pct: Optional[float] = None


class RecordStartRequest(BaseModel):
    motion_name: str
    mode: str = "single"  


class CompareStartRequest(BaseModel):
    reference_name: str
    bezier_order: int = 6


class CompareFrameResponse(BaseModel):
    reps_detected: int
    current_rep_progress: float
    last_rep_curve_distance: float
    last_rep_amplitude_diff: float
    score: float
    direction_ok: bool
    direction_hint: str
    feedback: str


# state matrix helpers 

def frames_to_state_matrix(frames: list[dict]) -> np.ndarray:
    """returns X (T, 9): [roll, pitch, yaw, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]"""
    rows = [[f["roll"], f["pitch"], f["yaw"],
             f["acc_x"], f["acc_y"], f["acc_z"],
             f["gyro_x"], f["gyro_y"], f["gyro_z"]] for f in frames]
    return np.array(rows, dtype=float)


def frames_to_state_matrix_dual(frames: list[dict]) -> np.ndarray:
    """returns X (T, 18): secondary band's 9 columns, then main band's 9 """
    rows = [[f.get(c, 0.0) for c in DUAL_COLUMNS] for f in frames]
    return np.array(rows, dtype=float)


def combine_with_main(f: dict) -> dict:
    """
    merge a secondary-band frame with the most recently seen main-band frame
    into a combined dict carrying both bands' keys - used for dual-mode
    sessions only.
    """
    combined = dict(f)
    m = _latest_main_frame or {}
    combined["m_acc_x"] = m.get("acc_x", 0.0)
    combined["m_acc_y"] = m.get("acc_y", 0.0)
    combined["m_acc_z"] = m.get("acc_z", 1.0)
    combined["m_gyro_x"] = m.get("gyro_x", 0.0)
    combined["m_gyro_y"] = m.get("gyro_y", 0.0)
    combined["m_gyro_z"] = m.get("gyro_z", 0.0)
    combined["m_roll"] = m.get("roll", 0.0)
    combined["m_pitch"] = m.get("pitch", 0.0)
    combined["m_yaw"] = m.get("yaw", 0.0)
    return combined


def extract_compare_slice(X: np.ndarray, indices: list[int]) -> np.ndarray:
    return X[:, indices]


def zscore_normalise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-6] = 1.0         
    return (X - mu) / sigma, mu, sigma


def pca_first_component(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalise then SVD 
    works for any column count (9-dim single-band or 18-dim dual-band).
    returns: (z1, V1, mu, sigma) where:
      - z1: Projection onto first PC, shape (T)
      - V1: First principal component vector (for reuse during comparison)
      - mu, sigma: Z-score parameters (for consistency)
    """
    Xn, mu, sigma = zscore_normalise(X)
    Xc = Xn - Xn.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V1 = Vt[0]
    z1 = Xc @ V1
    return z1, V1, mu, sigma


def project_onto_pca(X: np.ndarray, V1: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    
    Xn = (X - mu) / sigma
    Xc = Xn - Xn.mean(axis=0)
    return Xc @ V1


def segment_repetitions(z1: np.ndarray,
                         alpha: float = 0.12,
                         min_factor: float = 0.4,
                         max_factor: float = 2.5) -> list[tuple[int, int]]:
    T = len(z1)
    if T < 10:
        return []

    win = min(11, max(5, (T // 10) * 2 + 1))
    if win % 2 == 0:
        win += 1
    smooth = savgol_filter(z1, window_length=win, polyorder=2)

    signal_range = float(np.max(smooth) - np.min(smooth))
    if signal_range < 1e-6:
        return []

    prominence = max(alpha * signal_range, 1e-6)
    peak_distance = max(3, T // 12)
    peaks, _ = find_peaks(smooth, prominence=prominence, distance=peak_distance)
    troughs, _ = find_peaks(-smooth, prominence=prominence, distance=peak_distance)

    extrema = sorted(
        [(int(i), "max") for i in peaks] + [(int(i), "min") for i in troughs],
        key=lambda item: item[0],
    )

    if len(extrema) < 3:
        dz = np.zeros(T)
        dz[1:-1] = (smooth[2:] - smooth[:-2]) / 2.0
        sign = np.sign(dz)

        extrema = []
        for i in range(1, T - 1):
            if sign[i - 1] > 0 and sign[i + 1] < 0:
                extrema.append((i, "max"))
            elif sign[i - 1] < 0 and sign[i + 1] > 0:
                extrema.append((i, "min"))

        amp_thresh = alpha * np.max(np.abs(smooth))
        extrema = [(i, t) for i, t in extrema if abs(smooth[i]) > amp_thresh]

    if len(extrema) < 3:
        return []

    segments = []
    i = 0
    while i < len(extrema) - 2:
        if (extrema[i][1] == "min"
                and extrema[i + 1][1] == "max"
                and extrema[i + 2][1] == "min"):
            segments.append((extrema[i][0], extrema[i + 2][0]))
            i += 2
        else:
            i += 1

    if not segments:
        return []

    durations = [e - s for s, e in segments]
    T_mean = float(np.mean(durations))
    return [(s, e) for s, e in segments
            if min_factor * T_mean < (e - s) < max_factor * T_mean]


# Bézier fitting

def bernstein_matrix(n_points: int, order: int) -> np.ndarray:
    u = np.linspace(0, 1, n_points)
    B = np.zeros((n_points, order + 1))
    for k in range(order + 1):
        B[:, k] = comb(order, k) * (1 - u) ** (order - k) * u ** k
    return B


def fit_bezier(segment: np.ndarray, order: int = 8, lam: float = 5e-2,
               mu: np.ndarray | None = None, sigma: np.ndarray | None = None) -> np.ndarray:
    T_j, D = segment.shape
    if T_j <= order + 1:
        raise ValueError(f"Segment too short ({T_j} frames) for Bézier order {order}")

    if mu is not None and sigma is not None:
        seg_norm = (segment - mu) / sigma
    else:
        seg_norm, _, _ = zscore_normalise(segment)

    if T_j >= 7:
        win = min(7, T_j - (1 - T_j % 2))
        if win % 2 == 0:
            win -= 1
        if win >= 5:
            seg_norm = savgol_filter(seg_norm, window_length=win, polyorder=2, axis=0)

    B = bernstein_matrix(T_j, order)
    BtB = B.T @ B + lam * np.eye(order + 1)
    return np.linalg.solve(BtB, B.T @ seg_norm)


def eval_bezier(P: np.ndarray, n: int = 200) -> np.ndarray:
    order = P.shape[0] - 1
    return bernstein_matrix(n, order) @ P

def best_phase_shift(c1: np.ndarray, c2: np.ndarray,
                     max_shift_ratio: float = 0.25) -> int:
    if len(c1) != len(c2):
        n = min(len(c1), len(c2))
        c1 = c1[:n]
        c2 = c2[:n]
    max_shift = max(0, int(len(c1) * max_shift_ratio))
    best_shift = 0
    best = float("inf")
    for shift in range(-max_shift, max_shift + 1):
        shifted = np.roll(c1, shift, axis=0)
        score = float(np.mean(np.linalg.norm(shifted - c2, axis=1)))
        if score < best:
            best = score
            best_shift = shift
    return best_shift

def best_phase_aligned_distance(c1: np.ndarray, c2: np.ndarray,
                                max_shift_ratio: float = 0.5) -> float:
 
    if len(c1) != len(c2):
        n = min(len(c1), len(c2))
        c1 = c1[:n]
        c2 = c2[:n]

    if len(c1) == 0:
        return 0.0

    max_shift = max(0, int(len(c1) * max_shift_ratio))
    best = float("inf")
    for shift in range(-max_shift, max_shift + 1):
        shifted = np.roll(c1, shift, axis=0)
        score = float(np.mean(np.linalg.norm(shifted - c2, axis=1)))
        if score < best:
            best = score
    return best


def curve_distance(P1: np.ndarray, P2: np.ndarray, n: int = 200) -> float:
    c1 = eval_bezier(P1, n)
    c2 = eval_bezier(P2, n)
    return best_phase_aligned_distance(c1, c2)


def amplitude_diff(P1: np.ndarray, P2: np.ndarray, n: int = 200) -> float:
    c1 = eval_bezier(P1, n)
    c2 = eval_bezier(P2, n)
    a1 = float(np.max(np.linalg.norm(c1 - c1.mean(axis=0), axis=1)))
    a2 = float(np.max(np.linalg.norm(c2 - c2.mean(axis=0), axis=1)))
    return abs(a1 - a2)


def reference_amplitude(P_ref: np.ndarray, n: int = 200) -> float:
    
    c = eval_bezier(P_ref, n)
    return float(np.max(np.linalg.norm(c - c.mean(axis=0), axis=1)))


# direction detection

def dominant_angle_axis(X: np.ndarray) -> str:
    ranges = {
        "roll":  float(X[:, 0].max() - X[:, 0].min()),
        "pitch": float(X[:, 1].max() - X[:, 1].min()),
    }
    return max(ranges, key=ranges.__getitem__)


def check_direction(X_live_seg: np.ndarray,
                    ref_dominant_axis: str) -> tuple[bool, str]:
    
    live_dom = dominant_angle_axis(X_live_seg)
    if live_dom == ref_dominant_axis:
        return True, f"Correct direction ({live_dom} dominant)"

    labels = {
        "roll":  "forearm rotation / up-down curl (roll)",
        "pitch": "forward/backward flexion (pitch)",
        "yaw":   "side-to-side swing (yaw)",
    }
    hint = (
        f"Wrong movement direction! "
        f"Reference uses {labels.get(ref_dominant_axis, ref_dominant_axis)}, "
        f"but your rep used {labels.get(live_dom, live_dom)}. "
        f"Make sure you are curling the arm in the correct plane."
    )
    return False, hint


def assess_live_direction(X_full: np.ndarray,
                          ref_dominant_axis: str,
                          min_range: float = 6.0,
                          window: int = 80) -> tuple[bool, str]:
 
    if X_full is None or len(X_full) < 8:
        return True, ""

    recent = X_full[-min(window, len(X_full)):]
    angle_range = float(np.max(recent[:, :2]) - np.min(recent[:, :2]))
    if angle_range < min_range:
        return True, ""

    return check_direction(recent, ref_dominant_axis)


def motion_score(d_curve: float, delta_A: float, ref_amp: float,
                  tolerance: float = 0.6) -> float:
  
    if ref_amp < 1e-6:
        return 0.0
    d_ratio  = max(0.0, (d_curve  / ref_amp) - tolerance)
    dA_ratio = max(0.0, (delta_A / ref_amp) - tolerance)
    penalty = d_ratio * 0.50 + dA_ratio * 0.50   
    return round(float(max(0.0, min(100.0, 100.0 * (1.0 - penalty)))), 1)


def feedback_from_metrics(d_curve: float, delta_A: float,
                           ref_amp: float, direction_ok: bool,
                           direction_hint: str, score: float) -> str:
    if not direction_ok:
        return direction_hint

    if ref_amp < 1e-6:
        return "Cannot evaluate — reference amplitude is zero."
    if score >= 75:
        return "Great form! Very close to reference."
    
    shape_ratio = d_curve / ref_amp
    amp_ratio   = delta_A / ref_amp


    hints = []
    if shape_ratio > 0.30:
        hints.append("movement path deviates significantly from reference")
    elif shape_ratio > 0.15:
        hints.append("slight deviation in movement path")

    if amp_ratio > 0.30:
        hints.append("range of motion is off — check depth / height of movement")
    elif amp_ratio > 0.15:
        hints.append("range of motion slightly differs from reference")

    return (" ! " + "; ".join(hints)) if hints else "Acceptable form."


# UI signal helpers — single-band (9-dim) 

def frame_signals_from_matrix(X: np.ndarray) -> dict[str, list[float]]:
    empty: list[float] = []
    if X.size == 0:
        return {k: empty for k in ["roll", "pitch", "yaw",
                                    "acc_x", "acc_y", "acc_z",
                                    "gyro_x", "gyro_y", "gyro_z",
                                    "acc_mag", "gyro_mag"]}
    names = ["roll", "pitch", "yaw", "acc_x", "acc_y", "acc_z",
             "gyro_x", "gyro_y", "gyro_z"]
    out: dict[str, list[float]] = {n: X[:, i].tolist() for i, n in enumerate(names)}
    out["acc_mag"]  = np.linalg.norm(X[:, 3:6], axis=1).tolist()
    out["gyro_mag"] = np.linalg.norm(X[:, 6:9], axis=1).tolist()
    return out


def frame_signals_from_frames(frames: list[dict],
                               limit: int = UI_SIGNAL_WINDOW
                               ) -> dict[str, list[float]]:
    if not frames:
        return frame_signals_from_matrix(np.empty((0, 9)))
    
    return frame_signals_from_matrix(frames_to_state_matrix(frames[-limit:]))


# UI signal helpers — dual-band (18-dim) 

def frame_signals_from_matrix_dual(X: np.ndarray) -> dict[str, list[float]]:
    all_keys = DUAL_COLUMNS + ["acc_mag", "gyro_mag", "m_acc_mag", "m_gyro_mag"]
    if X.size == 0:
        return {k: [] for k in all_keys}
    out: dict[str, list[float]] = {n: X[:, i].tolist() for i, n in enumerate(DUAL_COLUMNS)}
    out["acc_mag"]    = np.linalg.norm(X[:, 3:6], axis=1).tolist()
    out["gyro_mag"]   = np.linalg.norm(X[:, 6:9], axis=1).tolist()
    out["m_acc_mag"]  = np.linalg.norm(X[:, 12:15], axis=1).tolist()
    out["m_gyro_mag"] = np.linalg.norm(X[:, 15:18], axis=1).tolist()
    return out


def frame_signals_from_frames_dual(frames: list[dict],
                                    limit: int = UI_SIGNAL_WINDOW
                                    ) -> dict[str, list[float]]:
    if not frames:
        return frame_signals_from_matrix_dual(np.empty((0, 18)))
    return frame_signals_from_matrix_dual(frames_to_state_matrix_dual(frames[-limit:]))


def resample_state_matrix(X: np.ndarray, n: int = UI_SIGNAL_WINDOW) -> np.ndarray:
    if X is None or len(X) == 0:
        return np.empty((0, X.shape[1] if X is not None and X.ndim == 2 else 9))
    if len(X) == 1:
        return np.repeat(X, n, axis=0)

    src = np.linspace(0.0, 1.0, len(X))
    dst = np.linspace(0.0, 1.0, n)
    out = np.zeros((n, X.shape[1]), dtype=float)
    for col in range(X.shape[1]):
        out[:, col] = np.interp(dst, src, X[:, col])
    return out


def frame_signals_from_resampled_matrix(X: np.ndarray,
                                        n: int = UI_SIGNAL_WINDOW
                                        ) -> dict[str, list[float]]:
    return frame_signals_from_matrix(resample_state_matrix(X, n=n))


def frame_signals_from_resampled_matrix_dual(X: np.ndarray,
                                             n: int = UI_SIGNAL_WINDOW
                                             ) -> dict[str, list[float]]:
    return frame_signals_from_matrix_dual(resample_state_matrix(X, n=n))


def frame_signals_from_control_points(cp: np.ndarray,
                                       n: int = UI_SIGNAL_WINDOW
                                       ) -> dict[str, list[float]]:
    """Evaluate reference Bézier (normalised space) and rebuild full 9-col matrix."""
    if cp is None or len(cp) == 0:
        return frame_signals_from_matrix(np.empty((0, 9)))
    curve = eval_bezier(cp, n=n)   # (n, 8) — normalised, yaw absent
    full = np.zeros((n, 9))
    for j, src_col in enumerate(COMPARE_INDICES):
        full[:, src_col] = curve[:, j]
    return frame_signals_from_matrix(full)


def frame_signals_from_control_points_dual(cp: np.ndarray,
                                            n: int = UI_SIGNAL_WINDOW
                                            ) -> dict[str, list[float]]:
    """Evaluate reference Bézier (normalised space) and rebuild full 18-col matrix."""
    if cp is None or len(cp) == 0:
        return frame_signals_from_matrix_dual(np.empty((0, 18)))
    curve = eval_bezier(cp, n=n)   # (n, 16) — normalised, both yaws absent
    full = np.zeros((n, 18))
    for j, src_col in enumerate(COMPARE_INDICES_DUAL):
        full[:, src_col] = curve[:, j]
    return frame_signals_from_matrix_dual(full)


# persistence

def _serialise(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(i) for i in obj]
    return obj


def save_motion(name: str, data: dict):
    (DATA_DIR / f"{name}.json").write_text(json.dumps(_serialise(data), indent=2))


def load_motion(name: str) -> dict:
    path = DATA_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"Motion '{name}' not found")
    data = json.loads(path.read_text())
    if "control_points" in data:
        data["control_points"] = np.array(data["control_points"])
    if "segments_control_points" in data:
        data["segments_control_points"] = [np.array(p)
                                            for p in data["segments_control_points"]]
    if "reference_plot_signals" in data:
        data["reference_plot_signals"] = data["reference_plot_signals"]
    #convert PCA parameters back to numpy arrays
    if "pca_v1" in data:
        data["pca_v1"] = np.array(data["pca_v1"])
    if "pca_mu" in data:
        data["pca_mu"] = np.array(data["pca_mu"])
    if "pca_sigma" in data:
        data["pca_sigma"] = np.array(data["pca_sigma"])
    
    if "cmp_mu" in data:
        data["cmp_mu"] = np.array(data["cmp_mu"])
    if "cmp_sigma" in data:
        data["cmp_sigma"] = np.array(data["cmp_sigma"])
    data.setdefault("mode", "single")  # old files predate dual mode
    return data


def list_motions() -> list[str]:
    return [p.stem for p in sorted(DATA_DIR.glob("*.json"))]


# Recording endpoints !!!!

@app.post("/record/start")
def record_start(req: RecordStartRequest):
    if req.mode not in ("single", "dual"):
        raise HTTPException(400, "mode must be 'single' or 'dual'")
    global _current_mode
    _current_mode = req.mode
    _record_session.clear()
    _record_session.update({
        "name": req.motion_name,
        "mode": req.mode,
        "frames": [],
        "started_at": time.time(),
    })
    return {"status": "recording", "motion_name": req.motion_name, "mode": req.mode}


@app.post("/frame")
def handle_frame(frame: IMUFrame):
    """Unified frame endpoint for the SECONDARY band — routes to the active
    session automatically. In dual mode, each frame is merged with the most
    recently seen main-band frame before being stored/analysed."""
    f = frame.model_dump()
    #print(f"[FRAME DEBUG] keys={list(f.keys())} batt_v={f.get('batt_v')} batt_pct={f.get('batt_pct')}")
    if f["host_timestamp"] is None:
        f["host_timestamp"] = time.time()

    if f.get("batt_v") is not None or f.get("batt_pct") is not None:
        _battery_state.update({
            "voltage": f.get("batt_v"),
            "percent": f.get("batt_pct"),
            "last_update": time.time(),
        })
        #print(f"[BATTERY] updated: v={f.get('batt_v')} pct={f.get('batt_pct')}")
    # strip before storing bc battery voltage doesn't belong in the motion
    f.pop("batt_v", None)
    f.pop("batt_pct", None)

    if _record_session:
        stored = combine_with_main(f) if _record_session.get("mode") == "dual" else f
        _record_session["frames"].append(stored)
        return {"mode": "recording",
                "frames_collected": len(_record_session["frames"])}

    if _compare_session:
        stored = combine_with_main(f) if _compare_session.get("mode") == "dual" else f
        return _process_compare_frame(stored)

    raise HTTPException(400,
        "No active session. Call /record/start or /compare/start first.")


@app.post("/frame/main")
def handle_main_frame(frame: IMUFrame):
    global _latest_main_frame
    f = frame.model_dump()
    if f["host_timestamp"] is None:
        f["host_timestamp"] = time.time()

    _latest_main_frame = f
    _main_frame_buffer.append(f)
    if len(_main_frame_buffer) > MAIN_FRAME_BUFFER_MAX:
        del _main_frame_buffer[: len(_main_frame_buffer) - MAIN_FRAME_BUFFER_MAX]

    return {"frames_buffered": len(_main_frame_buffer)}


@app.get("/main/state")
def main_state():
    """Live main-band signal preview — independent of record/compare mode."""
    return {
        "frames_count": len(_main_frame_buffer),
        "last_frame": _main_frame_buffer[-1] if _main_frame_buffer else None,
        "signals": frame_signals_from_frames(_main_frame_buffer),
    }


def _process_compare_frame(f: dict) -> CompareFrameResponse:
    _compare_session["frames"].append(f)
    frames = _compare_session["frames"]
    n = len(frames)

    mode = _compare_session.get("mode", "single")
    matrix_fn = frames_to_state_matrix_dual if mode == "dual" else frames_to_state_matrix
    compare_indices = COMPARE_INDICES_DUAL if mode == "dual" else COMPARE_INDICES

    if n < 30:
        _compare_session["current_rep_progress"] = 0.0
        return CompareFrameResponse(
            reps_detected=0,
            current_rep_progress=0.0,
            last_rep_curve_distance=0.0,
            last_rep_amplitude_diff=0.0,
            score=0.0,
            direction_ok=True,
            direction_hint="Collecting initial frames…",
            feedback="Collecting initial frames…",
        )

    recent_frames = frames[-COMPARE_ANALYSIS_WINDOW:]
    window_offset = n - len(recent_frames)
    X_full   = matrix_fn(recent_frames)
    X_compare_live = extract_compare_slice(X_full, compare_indices)

    pca_v1 = _compare_session.get("pca_v1")
    pca_mu = _compare_session.get("pca_mu")
    pca_sigma = _compare_session.get("pca_sigma")

    if pca_v1 is not None and len(pca_v1) > 0:
        z1 = project_onto_pca(X_compare_live, pca_v1, pca_mu, pca_sigma)
    else:
        z1, _, _, _ = pca_first_component(X_compare_live)

    segments = segment_repetitions(z1)
    segments = [(s + window_offset, e + window_offset) for s, e in segments]

    ref_dom = _compare_session.get("ref_dominant_axis", "roll")
    live_dir_ok, live_dir_hint = assess_live_direction(X_full, ref_dom)
    _compare_session["last_direction_ok"] = bool(live_dir_ok)
    _compare_session["last_direction_hint"] = live_dir_hint if live_dir_hint else _compare_session.get("last_direction_hint", "")

    known_reps = len(_compare_session["completed_reps"])
    last_processed_end = int(_compare_session.get("last_processed_segment_end", -1))
    new_segments = [seg for seg in segments if seg[1] > last_processed_end]
    if new_segments:
        for s, e in new_segments:
            local_s = max(0, s - window_offset)
            local_e = max(0, e - window_offset)
            seg_raw  = X_full[local_s:local_e + 1]
            seg_data = extract_compare_slice(seg_raw, compare_indices)
            order    = _compare_session["bezier_order"]
            if len(seg_data) < order + 2:
                continue
            try:
                cmp_mu = _compare_session.get("cmp_mu")
                cmp_sigma = _compare_session.get("cmp_sigma")
                if cmp_mu is not None and len(cmp_mu) > 0:
                    P_j = fit_bezier(seg_data, order=order, mu=cmp_mu, sigma=cmp_sigma)
                else:
                    P_j = fit_bezier(seg_data, order=order)
                ref_P  = _compare_session["ref_P"]
                min_cp = min(P_j.shape[0], ref_P.shape[0])
                min_d  = min(P_j.shape[1], ref_P.shape[1])
                d      = curve_distance(P_j[:min_cp, :min_d], ref_P[:min_cp, :min_d])
                dA     = amplitude_diff(P_j[:min_cp, :min_d], ref_P[:min_cp, :min_d])

                ref_amp  = _compare_session["ref_amp"]
                dir_ok, dir_hint = check_direction(seg_raw, ref_dom)
                score    = motion_score(d, dA, ref_amp)
                fb       = feedback_from_metrics(d, dA, ref_amp, dir_ok, dir_hint, score)

                _compare_session["completed_reps"].append({
                    "rep": len(_compare_session["completed_reps"]) + 1,
                    "d_curve": d, "delta_A": dA,
                    "score": score,
                    "direction_ok": dir_ok,
                    "direction_hint": dir_hint,
                    "feedback": fb,
                })
                _compare_session.update({
                    "last_d_curve": d, "last_delta_A": dA,
                    "last_score": score, "last_feedback": fb,
                    "last_direction_ok": dir_ok,
                    "last_direction_hint": dir_hint,
                    "last_processed_segment_end": e,
                })
            except Exception:
                pass

    rep_progress = 0.0
    if segments:
        last_end = segments[-1][1]
        partial  = n - 1 - last_end
        avg_rep  = float(np.mean([e - s for s, e in segments]))
        rep_progress = min(1.0, partial / max(avg_rep, 1.0))
    _compare_session["current_rep_progress"] = rep_progress

    if known_reps == 0 and live_dir_hint:
        _compare_session["last_feedback"] = live_dir_hint

    return CompareFrameResponse(
        reps_detected=len(_compare_session["completed_reps"]),
        current_rep_progress=round(rep_progress, 3),
        last_rep_curve_distance=round(float(_compare_session.get("last_d_curve", 0.0)), 4),
        last_rep_amplitude_diff=round(float(_compare_session.get("last_delta_A", 0.0)), 4),
        score=round(float(_compare_session.get("last_score", 0.0)), 1),
        direction_ok=bool(_compare_session.get("last_direction_ok", True)),
        direction_hint=str(_compare_session.get("last_direction_hint", "")),
        feedback=str(_compare_session.get("last_feedback", live_dir_hint or "Waiting for data…")),
    )


@app.post("/record/stop")
def record_stop(bezier_order: int = 8):
    if not _record_session:
        raise HTTPException(400, "No active recording.")

    frames = _record_session["frames"]
    name   = _record_session["name"]
    mode   = _record_session.get("mode", "single")
    _record_session.clear()
    global _current_mode
    _current_mode = "single" 

    if len(frames) < 20:
        raise HTTPException(422, "Too few frames (need ≥ 20).")

    if mode == "dual":
        X_full = frames_to_state_matrix_dual(frames)
        compare_indices = COMPARE_INDICES_DUAL
        resampled_signals_fn = frame_signals_from_resampled_matrix_dual
    else:
        X_full = frames_to_state_matrix(frames)
        compare_indices = COMPARE_INDICES
        resampled_signals_fn = frame_signals_from_resampled_matrix

    X_compare = extract_compare_slice(X_full, compare_indices)

    # was: pca_first_component(X_full)
    z1, V1, mu_z, sigma_z = pca_first_component(X_compare)
    Xn_cmp, mu_cmp, sigma_cmp = zscore_normalise(X_compare)
    segments = segment_repetitions(z1)

    if not segments:
        segments = [(0, len(frames) - 1)]   # whole recording = one rep

    seg_cps = []
    for s, e in segments:
        seg = X_compare[s:e + 1]
        if len(seg) < bezier_order + 2:
            continue
        try:
            seg_cps.append(fit_bezier(seg, order=bezier_order, mu=mu_cmp, sigma=sigma_cmp))
        except ValueError:
            pass

    if not seg_cps:
        raise HTTPException(422,
            "Could not fit Bézier segments — try recording more repetitions.")

    if len(seg_cps) > 1:
        n_eval = 200
        curves = [eval_bezier(P, n=n_eval) for P in seg_cps]
        anchor = curves[0]
        aligned = [anchor]
        for c in curves[1:]:
            shift = best_phase_shift(c, anchor)
            aligned.append(np.roll(c, shift, axis=0))
        mean_curve = np.mean(aligned, axis=0)

        B = bernstein_matrix(n_eval, bezier_order)
        BtB = B.T @ B + 1e-4 * np.eye(bezier_order + 1)
        ref_P = np.linalg.solve(BtB, B.T @ mean_curve)
    else:
        ref_P = seg_cps[0]

    ref_amp = reference_amplitude(ref_P)   

    # Determine the dominant axis per-rep and take the majority. Using the
    # whole raw recording (which includes rest periods between reps, minor
    # drift while repositioning, etc.) can report a different axis than any
    # individual rep actually shows — and live checks only ever see one rep
    # at a time, so the reference needs to be computed on the same scope.
    seg_axes = []
    for s, e in segments:
        seg = X_full[s:e + 1]
        if len(seg) >= 4:
            seg_axes.append(dominant_angle_axis(seg))
    dom_axis = max(set(seg_axes), key=seg_axes.count) if seg_axes else dominant_angle_axis(X_full)
    reference_plot_signals = resampled_signals_fn(X_full)

    save_motion(name, {
        "name": name,
        "mode": mode,
        "bezier_order": bezier_order,
        "n_reps": len(seg_cps),
        "control_points": ref_P,
        "segments_control_points": seg_cps,
        "n_frames": len(frames),
        "state_dim": X_compare.shape[1],
        "ref_amplitude": float(ref_amp),
        "dominant_axis": dom_axis,
        "reference_plot_signals": reference_plot_signals,
        "pca_v1": V1.tolist(),  # Store the PCA direction vector
        "pca_mu": mu_z.tolist(),  # Store z-score mean
        "pca_sigma": sigma_z.tolist(),  # Store z-score std
        "recorded_at": time.time(),
        "cmp_mu": mu_cmp.tolist(),
        "cmp_sigma": sigma_cmp.tolist(),
    })

    return {
        "status": "saved",
        "motion_name": name,
        "mode": mode,
        "reps_detected": len(seg_cps),
        "n_frames": len(frames),
        "dominant_axis": dom_axis,
        "ref_amplitude": round(float(ref_amp), 3),
    }


# motion management 

@app.get("/motions")
def get_motions():
    return {"motions": list_motions()}


@app.delete("/motions/{name}")
def delete_motion(name: str):
    path = DATA_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"Motion '{name}' not found")
    path.unlink()
    return {"deleted": name}


@app.get("/motions/{name}")
def get_motion_info(name: str):
    d = load_motion(name)
    return {
        "name": d["name"],
        "mode": d.get("mode", "single"),
        "bezier_order": d["bezier_order"],
        "n_reps_in_reference": d["n_reps"],
        "n_frames": d["n_frames"],
        "state_dim": d["state_dim"],
        "ref_amplitude": d.get("ref_amplitude"),
        "dominant_axis": d.get("dominant_axis"),
        "recorded_at": d.get("recorded_at"),
    }


# comparison endpoints 

@app.post("/compare/start")
def compare_start(req: CompareStartRequest):
    ref = load_motion(req.reference_name)
    mode = ref.get("mode", "single")
    ref_P   = ref["control_points"]
    ref_amp = float(ref.get("ref_amplitude") or reference_amplitude(ref_P))
    global _current_mode
    _current_mode = mode

    # load PCA parameters from reference for consistent segmentation
    pca_v1 = np.array(ref.get("pca_v1", []))
    pca_mu = np.array(ref.get("pca_mu", []))
    pca_sigma = np.array(ref.get("pca_sigma", []))
    cmp_mu = np.array(ref.get("cmp_mu", []))
    cmp_sigma = np.array(ref.get("cmp_sigma", []))

    fallback_signals_fn = frame_signals_from_control_points_dual if mode == "dual" else frame_signals_from_control_points

    _compare_session.clear()
    _compare_session.update({
        "reference_name": req.reference_name,
        "mode": mode,
        "ref_P": ref_P,
        "ref_amp": ref_amp,
        "ref_dominant_axis": ref.get("dominant_axis", "roll"),
        "ref_signals": ref.get("reference_plot_signals") or fallback_signals_fn(ref_P),
        "bezier_order": req.bezier_order,
        "pca_v1": pca_v1,  # Store reference PCA direction
        "pca_mu": pca_mu,  # Store reference z-score mean
        "pca_sigma": pca_sigma,  # Store reference z-score std
        "cmp_mu": cmp_mu,
        "cmp_sigma": cmp_sigma,
        "frames": [],
        "completed_reps": [],
        "started_at": time.time(),
        "last_d_curve": 0.0, "last_delta_A": 0.0,
        "last_score": 0.0, "current_rep_progress": 0.0,
        "last_direction_ok": True, "last_direction_hint": "",
        "last_feedback": "Waiting for data…",
    })
    return {
        "status": "comparing",
        "reference": req.reference_name,
        "mode": mode,
        "ref_amplitude": ref_amp,
        "dominant_axis": _compare_session["ref_dominant_axis"],
    }


@app.post("/compare/stop")
def compare_stop():
    if not _compare_session:
        raise HTTPException(400, "No active comparison.")

    reps     = list(_compare_session["completed_reps"])
    ref_name = _compare_session["reference_name"]
    n_frames = len(_compare_session["frames"])
    _compare_session.clear()
    global _current_mode
    _current_mode = "single"

    if not reps:
        return {
            "reference": ref_name,
            "reps_analysed": 0,
            "n_frames": n_frames,
            "summary": "No complete repetitions detected.",
            "avg_score": 0.0, "reps": [],
        }

    avg_score = float(np.mean([r.get("score", 0.0) for r in reps]))
    return {
        "reference": ref_name,
        "reps_analysed": len(reps),
        "n_frames": n_frames,
        "avg_curve_distance": round(float(np.mean([r["d_curve"] for r in reps])), 4),
        "avg_amplitude_diff": round(float(np.mean([r["delta_A"] for r in reps])), 4),
        "avg_score": round(avg_score, 1),
        "overall_feedback": reps[-1]["feedback"] if reps else "No reps.",
        "reps": reps,
    }


# state polling for the UI 

#uncomment 
@app.get("/record/state")
def record_state():
    if not _record_session:
        return {
            "active": False, "motion_name": None, "mode": "single", "frames_count": 0,
            "elapsed_seconds": 0.0, "last_frame": None,
            "signals": frame_signals_from_frames([]),
            "status_text": "No active recording.", "reps_detected": 0,
        }
    frames = _record_session["frames"]
    mode = _record_session.get("mode", "single")
    signals_fn = frame_signals_from_frames_dual if mode == "dual" else frame_signals_from_frames
    return {
        "active": True,
        "motion_name": _record_session.get("name"),
        "mode": mode,
        "frames_count": len(frames),
        "elapsed_seconds": round(time.time() - float(_record_session["started_at"]), 1),
        "last_frame": frames[-1] if frames else None,
        "signals": signals_fn(frames),
        "status_text": f"Recording {len(frames)} frames…",
        "reps_detected": 0,
    }


@app.get("/compare/state")
def compare_state():
    if not _compare_session:
        empty = frame_signals_from_frames([])
        return {
            "active": False, "reference_name": None, "mode": "single", "bezier_order": None,
            "frames_count": 0, "elapsed_seconds": 0.0,
            "reps_detected": 0, "current_rep_progress": 0.0,
            "last_rep_curve_distance": 0.0, "last_rep_amplitude_diff": 0.0,
            "score": 0.0, "direction_ok": True, "direction_hint": "",
            "feedback": "No active comparison.",
            "live_signals": empty, "reference_signals": empty,
            "completed_reps": [], "last_frame": None,
        }
    frames = _compare_session["frames"]
    mode = _compare_session.get("mode", "single")
    signals_fn = frame_signals_from_frames_dual if mode == "dual" else frame_signals_from_frames
    empty = frame_signals_from_frames_dual([]) if mode == "dual" else frame_signals_from_frames([])
    return {
        "active": True,
        "reference_name": _compare_session.get("reference_name"),
        "mode": mode,
        "bezier_order": _compare_session.get("bezier_order"),
        "frames_count": len(frames),
        "elapsed_seconds": round(time.time() - float(_compare_session["started_at"]), 1),
        "reps_detected": len(_compare_session.get("completed_reps", [])),
        "current_rep_progress": round(float(_compare_session.get("current_rep_progress", 0.0)), 3),
        "last_rep_curve_distance": round(float(_compare_session.get("last_d_curve", 0.0)), 4),
        "last_rep_amplitude_diff": round(float(_compare_session.get("last_delta_A", 0.0)), 4),
        "score": round(float(_compare_session.get("last_score", 0.0)), 1),
        "direction_ok": bool(_compare_session.get("last_direction_ok", True)),
        "direction_hint": str(_compare_session.get("last_direction_hint", "")),
        "feedback": str(_compare_session.get("last_feedback", "Waiting for data…")),
        "live_signals": signals_fn(frames),
        "reference_signals": _compare_session.get("ref_signals") or empty,
        "completed_reps": _compare_session.get("completed_reps", []),
        "last_frame": frames[-1] if frames else None,
    }


# band connection status 
 
@app.post("/link/status")
def update_link_status(status: LinkStatus):
    _link_state.update({
        "secondary_connected": status.secondary_connected,
        "main_connected": status.main_connected,
        "state": status.state,
        "last_update": status.timestamp or time.time(),
    })
    return {"ok": True}

@app.get("/battery/status")
def get_battery_status():
    last = _battery_state.get("last_update")
    stale = last is None or (time.time() - last) > BATTERY_STALE_SECONDS
    return {
        "available": last is not None and not stale,
        "voltage": _battery_state.get("voltage"),
        "percent": _battery_state.get("percent"),
        "stale": stale,
        "last_update": last,
    }

@app.get("/mode/current")
def get_current_mode():
    return {"mode": _current_mode}
 
@app.get("/link/status")
def get_link_status():
    last = _link_state.get("last_update")
    stale = last is None or (time.time() - last) > LINK_STALE_SECONDS
    return {
        "secondary_connected": bool(_link_state["secondary_connected"]) and not stale,
        "main_connected": bool(_link_state["main_connected"]) and not stale,
        "state": _link_state.get("state", "unknown"),
        "stale": stale,
        "last_update": last,
    }
 
 
# Health / UI 

@app.get("/health")
def health():
    return {"service": "Twyst Backend", "version": "0.4.0",
            "status": "ok", "saved_motions": list_motions()}


@app.get("/", response_class=HTMLResponse)
def root():
    ui = Path(__file__).with_name("ui.html")
    if ui.exists():
        return ui.read_text(encoding="utf-8")
    return HTMLResponse(
        "<h1>Twyst Backend v0.2</h1><p>Place ui.html next to main.py</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)