import json
import time
from pathlib import Path
from typing import Optional

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

app = FastAPI(title="Twyst Backend", version="0.3.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_record_session: dict = {}
_compare_session: dict = {}

UI_SIGNAL_WINDOW = 200
COMPARE_ANALYSIS_WINDOW = 240

# Band connection tracking.
# The backend never talks to either ESP32 directly — the secondary band is
# wired over USB to esp32_bridge.py, which relays a periodic "LINK" line
# from the firmware here. secondary_connected means "the bridge is alive
# and reading from the secondary band"; main_connected means "the secondary
# band's BLE link to the main band is currently up".
_link_state: dict = {
    "secondary_connected": False,
    "main_connected": False,
    "state": "unknown",
    "last_update": None,
}
LINK_STALE_SECONDS = 5.0  # if the bridge hasn't posted in this long, treat as unknown/offline
 
 
class LinkStatus(BaseModel):
    secondary_connected: bool
    main_connected: bool
    state: str = "unknown"
    timestamp: Optional[float] = None


# column layout of the 9-dim state vector
# [roll, pitch, yaw, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
COMPARE_INDICES = [0, 1, 3, 4, 5, 6, 7, 8]   # drops col-2 (yaw, always 0)


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
    timestamp: Optional[float] = None


class RecordStartRequest(BaseModel):
    motion_name: str


class CompareStartRequest(BaseModel):
    reference_name: str
    bezier_order: int = 8


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
    """Returns X (T, 9): [roll, pitch, yaw, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]."""
    rows = [[f["roll"], f["pitch"], f["yaw"],
             f["acc_x"], f["acc_y"], f["acc_z"],
             f["gyro_x"], f["gyro_y"], f["gyro_z"]] for f in frames]
    return np.array(rows, dtype=float)


def extract_compare_slice(X: np.ndarray) -> np.ndarray:
    """Drop yaw column (always 0 on MPU9265). Returns (T, 8)."""
    return X[:, COMPARE_INDICES]


def zscore_normalise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalise columns.  Crucial: without this, gyro values (range ~100 deg/s)
    dominate PCA over angle values (range ~90°) by a factor of ~10x, causing the
    first principal component to track gyro noise instead of the actual motion cycle.
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-6] = 1.0          # prevent divide-by-zero on constant columns
    return (X - mu) / sigma, mu, sigma


# PCA-based repetition segmentation (§3.2) 

def pca_first_component(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalise then SVD → first PC projection z1 and the PCA component vector.
    Returns: (z1, V1, mu, sigma) where:
      - z1: Projection onto first PC, shape (T,)
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
    """
    Project data X onto a pre-computed PCA direction V1.
    Use the same z-score parameters (mu, sigma) as the reference.
    Returns z1 signal, shape (T,).
    """
    Xn = (X - mu) / sigma
    Xc = Xn - Xn.mean(axis=0)
    return Xc @ V1


def segment_repetitions(z1: np.ndarray,
                         alpha: float = 0.12,
                         min_factor: float = 0.4,
                         max_factor: float = 2.5) -> list[tuple[int, int]]:
    """
    Segment reps by finding trough→peak→trough cycles in the first PC signal.
    This is more stable for live streams than derivative sign flips and can
    separate multiple curls in one comparison stream.
    """
    T = len(z1)
    if T < 10:
        return []

    win = min(21, max(5, (T // 4) * 2 + 1))
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
        # fallback for flatter / shorter sequences where peak picking is too strict
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


# Bézier fitting (§3.3)

def bernstein_matrix(n_points: int, order: int) -> np.ndarray:
    u = np.linspace(0, 1, n_points)
    B = np.zeros((n_points, order + 1))
    for k in range(order + 1):
        B[:, k] = comb(order, k) * (1 - u) ** (order - k) * u ** k
    return B


def fit_bezier(segment: np.ndarray, order: int = 8,
               lam: float = 1e-4) -> np.ndarray:
    """
    Fit a Bézier curve to `segment` (T_j, D).
    The data is z-score normalised inside this function so that different
    physical units (degrees, g, deg/s) sit on an equal scale.  Control
    points are returned in the normalised space; comparison also happens
    in normalised space so the units cancel out.
    """
    T_j, D = segment.shape
    if T_j <= order + 1:
        raise ValueError(f"Segment too short ({T_j} frames) for Bézier order {order}")
    seg_norm, _, _ = zscore_normalise(segment)
    B = bernstein_matrix(T_j, order)
    BtB = B.T @ B + lam * np.eye(order + 1)
    return np.linalg.solve(BtB, B.T @ seg_norm)   # (order+1, D) in normalised space


def eval_bezier(P: np.ndarray, n: int = 200) -> np.ndarray:
    order = P.shape[0] - 1
    return bernstein_matrix(n, order) @ P


def best_phase_aligned_distance(c1: np.ndarray, c2: np.ndarray,
                                max_shift_ratio: float = 0.25) -> float:
    """
    Return the mean L2 distance after searching for the best temporal shift.
    This makes comparison robust when a rep starts later or earlier than the
    reference recording.
    """
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
    """d_curve: mean L2 distance between evaluated curves (eq. 1 of paper)."""
    c1 = eval_bezier(P1, n)
    c2 = eval_bezier(P2, n)
    return best_phase_aligned_distance(c1, c2)


def amplitude_diff(P1: np.ndarray, P2: np.ndarray, n: int = 200) -> float:
    """ΔA: difference in peak excursion from curve mean (eq. 3 of paper)."""
    c1 = eval_bezier(P1, n)
    c2 = eval_bezier(P2, n)
    a1 = float(np.max(np.linalg.norm(c1 - c1.mean(axis=0), axis=1)))
    a2 = float(np.max(np.linalg.norm(c2 - c2.mean(axis=0), axis=1)))
    return abs(a1 - a2)


def reference_amplitude(P_ref: np.ndarray, n: int = 200) -> float:
    """Peak excursion of the reference — used to normalise the score."""
    c = eval_bezier(P_ref, n)
    return float(np.max(np.linalg.norm(c - c.mean(axis=0), axis=1)))


# direction detection

def dominant_angle_axis(X: np.ndarray) -> str:
    """
    Which of roll/pitch/yaw carries the most range in this recording?
    For a correct bicep curl (wrist band, arm curling up-down) this should
    be 'roll' or 'pitch' depending on how the band is oriented.
    If the user swings the arm sideways instead, 'yaw' will dominate.
    """
    ranges = {
        "roll":  float(X[:, 0].max() - X[:, 0].min()),
        "pitch": float(X[:, 1].max() - X[:, 1].min()),
        "yaw":   float(X[:, 2].max() - X[:, 2].min()),
    }
    return max(ranges, key=ranges.__getitem__)


def check_direction(X_live_seg: np.ndarray,
                    ref_dominant_axis: str) -> tuple[bool, str]:
    """Return (direction_ok, human_readable_hint)."""
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
    """
    Check the dominant angle axis on the most recent raw frames.
    This is used for immediate feedback while a rep is still in progress.
    """
    if X_full is None or len(X_full) < 8:
        return True, ""

    recent = X_full[-min(window, len(X_full)):]
    angle_range = float(np.max(recent[:, :3]) - np.min(recent[:, :3]))
    if angle_range < min_range:
        return True, ""

    return check_direction(recent, ref_dominant_axis)


# Score & feedback 

def motion_score(d_curve: float, delta_A: float, ref_amp: float) -> float:
    """
    Normalise by the reference amplitude.
    - A rep identical to the reference → 100.
    - A rep whose shape deviates by one full amplitude → ≈ 0.
    d_curve contributes 70 % of the penalty; delta_A (range-of-motion) 30 %.
    """
    if ref_amp < 1e-6:
        return 0.0
    penalty = (d_curve / ref_amp) * 0.70 + (delta_A / ref_amp) * 0.30
    return round(float(max(0.0, min(100.0, 100.0 * (1.0 - penalty)))), 1)


def feedback_from_metrics(d_curve: float, delta_A: float,
                           ref_amp: float, direction_ok: bool,
                           direction_hint: str) -> str:
    if not direction_ok:
        return direction_hint

    if ref_amp < 1e-6:
        return "Cannot evaluate — reference amplitude is zero."

    shape_ratio = d_curve / ref_amp
    amp_ratio   = delta_A / ref_amp

    if shape_ratio < 0.08 and amp_ratio < 0.10:
        return "Great form! Very close to reference."

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


# UI signal helpers 

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


def resample_state_matrix(X: np.ndarray, n: int = UI_SIGNAL_WINDOW) -> np.ndarray:
    """Linearly resample a state matrix to n samples for UI display."""
    if X is None or len(X) == 0:
        return np.empty((0, 9))
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
    return data


def list_motions() -> list[str]:
    return [p.stem for p in sorted(DATA_DIR.glob("*.json"))]


# Recording endpoints !!!!

@app.post("/record/start")
def record_start(req: RecordStartRequest):
    _record_session.clear()
    _record_session.update({
        "name": req.motion_name,
        "frames": [],
        "started_at": time.time(),
    })
    return {"status": "recording", "motion_name": req.motion_name}


@app.post("/frame")
def handle_frame(frame: IMUFrame):
    """Unified frame endpoint — routes to the active session automatically."""
    f = frame.model_dump()
    f["timestamp"] = f["timestamp"] or time.time()

    if _record_session:
        _record_session["frames"].append(f)
        return {"mode": "recording",
                "frames_collected": len(_record_session["frames"])}

    if _compare_session:
        return _process_compare_frame(f)

    raise HTTPException(400,
        "No active session. Call /record/start or /compare/start first.")


def _process_compare_frame(f: dict) -> CompareFrameResponse:
    _compare_session["frames"].append(f)
    frames = _compare_session["frames"]
    n = len(frames)

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
    X_full   = frames_to_state_matrix(recent_frames)
    
    # Use reference PCA direction for consistent segmentation across all frames
    pca_v1 = _compare_session.get("pca_v1")
    pca_mu = _compare_session.get("pca_mu")
    pca_sigma = _compare_session.get("pca_sigma")
    
    if pca_v1 is not None and len(pca_v1) > 0:
        # Project live data onto reference PCA direction
        z1 = project_onto_pca(X_full, pca_v1, pca_mu, pca_sigma)
    else:
        # Fallback for old reference files without PCA data
        z1, _, _, _ = pca_first_component(X_full)
    
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
            seg_data = extract_compare_slice(seg_raw)
            order    = _compare_session["bezier_order"]
            if len(seg_data) < order + 2:
                continue
            try:
                P_j    = fit_bezier(seg_data, order=order)
                ref_P  = _compare_session["ref_P"]
                min_cp = min(P_j.shape[0], ref_P.shape[0])
                min_d  = min(P_j.shape[1], ref_P.shape[1])
                d      = curve_distance(P_j[:min_cp, :min_d], ref_P[:min_cp, :min_d])
                dA     = amplitude_diff(P_j[:min_cp, :min_d], ref_P[:min_cp, :min_d])

                ref_amp  = _compare_session["ref_amp"]
                dir_ok, dir_hint = check_direction(seg_raw, ref_dom)
                score    = motion_score(d, dA, ref_amp)
                fb       = feedback_from_metrics(d, dA, ref_amp, dir_ok, dir_hint)

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
    _record_session.clear()

    if len(frames) < 20:
        raise HTTPException(422, "Too few frames (need ≥ 20).")

    X_full    = frames_to_state_matrix(frames)
    X_compare = extract_compare_slice(X_full)

    z1, V1, mu_z, sigma_z = pca_first_component(X_full)
    segments = segment_repetitions(z1)

    if not segments:
        segments = [(0, len(frames) - 1)]   # whole recording = one rep

    seg_cps = []
    for s, e in segments:
        seg = X_compare[s:e + 1]
        if len(seg) < bezier_order + 2:
            continue
        try:
            seg_cps.append(fit_bezier(seg, order=bezier_order))
        except ValueError:
            pass

    if not seg_cps:
        raise HTTPException(422,
            "Could not fit Bézier segments — try recording more repetitions.")

    ref_P    = np.mean(seg_cps, axis=0)
    ref_amp  = reference_amplitude(ref_P)
    dom_axis = dominant_angle_axis(X_full)
    reference_plot_signals = frame_signals_from_resampled_matrix(X_full)

    save_motion(name, {
        "name": name,
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
    })

    return {
        "status": "saved",
        "motion_name": name,
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
    ref_P   = ref["control_points"]
    ref_amp = float(ref.get("ref_amplitude") or reference_amplitude(ref_P))
    
    # Load PCA parameters from reference for consistent segmentation
    pca_v1 = np.array(ref.get("pca_v1", []))
    pca_mu = np.array(ref.get("pca_mu", []))
    pca_sigma = np.array(ref.get("pca_sigma", []))

    _compare_session.clear()
    _compare_session.update({
        "reference_name": req.reference_name,
        "ref_P": ref_P,
        "ref_amp": ref_amp,
        "ref_dominant_axis": ref.get("dominant_axis", "roll"),
        "ref_signals": ref.get("reference_plot_signals") or frame_signals_from_control_points(ref_P),
        "bezier_order": req.bezier_order,
        "pca_v1": pca_v1,  # Store reference PCA direction
        "pca_mu": pca_mu,  # Store reference z-score mean
        "pca_sigma": pca_sigma,  # Store reference z-score std
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

@app.get("/record/state")
def record_state():
    if not _record_session:
        return {
            "active": False, "motion_name": None, "frames_count": 0,
            "elapsed_seconds": 0.0, "last_frame": None,
            "signals": frame_signals_from_frames([]),
            "status_text": "No active recording.", "reps_detected": 0,
        }
    frames = _record_session["frames"]
    return {
        "active": True,
        "motion_name": _record_session.get("name"),
        "frames_count": len(frames),
        "elapsed_seconds": round(time.time() - float(_record_session["started_at"]), 1),
        "last_frame": frames[-1] if frames else None,
        "signals": frame_signals_from_frames(frames),
        "status_text": f"Recording {len(frames)} frames…",
        "reps_detected": 0,
    }


@app.get("/compare/state")
def compare_state():
    if not _compare_session:
        empty = frame_signals_from_frames([])
        return {
            "active": False, "reference_name": None, "bezier_order": None,
            "frames_count": 0, "elapsed_seconds": 0.0,
            "reps_detected": 0, "current_rep_progress": 0.0,
            "last_rep_curve_distance": 0.0, "last_rep_amplitude_diff": 0.0,
            "score": 0.0, "direction_ok": True, "direction_hint": "",
            "feedback": "No active comparison.",
            "live_signals": empty, "reference_signals": empty,
            "completed_reps": [], "last_frame": None,
        }
    frames = _compare_session["frames"]
    return {
        "active": True,
        "reference_name": _compare_session.get("reference_name"),
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
        "live_signals": frame_signals_from_frames(frames),
        "reference_signals": _compare_session.get("ref_signals")
                             or frame_signals_from_frames([]),
        "completed_reps": _compare_session.get("completed_reps", []),
        "last_frame": frames[-1] if frames else None,
    }


# Band connection status 
 
@app.post("/link/status")
def update_link_status(status: LinkStatus):
    _link_state.update({
        "secondary_connected": status.secondary_connected,
        "main_connected": status.main_connected,
        "state": status.state,
        "last_update": status.timestamp or time.time(),
    })
    return {"ok": True}
 
 
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
    return {"service": "Twyst Backend", "version": "0.3.0",
            "status": "ok", "saved_motions": list_motions()}


@app.get("/", response_class=HTMLResponse)
def root():
    ui = Path(__file__).with_name("ui.html")
    if ui.exists():
        return ui.read_text(encoding="utf-8")
    return HTMLResponse(
        "<h1>Twyst Backend v0.2</h1><p>Place ui.html next to main.py</p>")