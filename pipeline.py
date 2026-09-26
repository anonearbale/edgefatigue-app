"""EdgeFatigue inference pipeline.

Mirrors EduFatigue_Watch_Pipeline.ipynb exactly: 20 s windows with a
10 s step at 100 Hz, the same IMU and HRV features, the saved
StandardScaler and feature list, and the 0.35/0.40/0.25 soft-vote
ensemble of Random Forest, XGBoost, and Gradient Boosting.
"""
import io
import os
import time
import zipfile

import joblib
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew

FS = 100
WINDOW_LEN = 20 * FS
STEP_LEN = 10 * FS
WEIGHTS = {"rf": 0.35, "xgb": 0.40, "gb": 0.25}
MODEL_FILES = {
    "rf": "watch_rf.pkl",
    "xgb": "watch_xgb.pkl",
    "gb": "watch_gb.pkl",
    "scaler": "watch_scaler.pkl",
    "features": "watch_features.pkl",
    "threshold": "watch_threshold.pkl",
}

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ── Features (identical to the training notebook) ────────────────────
def compute_imu_features(ax, ay, az, gx, gy, gz, fs=FS):
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    def stats(sig, prefix):
        return {
            f"{prefix}_mean": np.mean(sig),
            f"{prefix}_std": np.std(sig),
            f"{prefix}_rms": np.sqrt(np.mean(sig**2)),
            f"{prefix}_range": np.ptp(sig),
            f"{prefix}_skew": float(skew(sig)),
            f"{prefix}_kurt": float(kurtosis(sig)),
        }

    feats = {}
    for sig, name in [(ax, "acc_x"), (ay, "acc_y"), (az, "acc_z"), (acc_mag, "acc_mag"),
                      (gx, "gyro_x"), (gy, "gyro_y"), (gz, "gyro_z"), (gyro_mag, "gyro_mag")]:
        feats.update(stats(sig, name))

    jerk = np.diff(acc_mag) * fs
    feats.update({
        "jerk_mean": np.mean(np.abs(jerk)),
        "jerk_std": np.std(jerk),
        "jerk_rms": np.sqrt(np.mean(jerk**2)),
        "sma": (np.sum(np.abs(ax)) + np.sum(np.abs(ay)) + np.sum(np.abs(az))) / len(ax),
        "acc_corr_xy": np.corrcoef(ax, ay)[0, 1],
        "acc_corr_xz": np.corrcoef(ax, az)[0, 1],
        "acc_corr_yz": np.corrcoef(ay, az)[0, 1],
    })

    for sig, name in [(acc_mag, "acc"), (gyro_mag, "gyro")]:
        f, pxx = welch(sig, fs=fs, nperseg=min(256, len(sig)))
        p = pxx / (np.sum(pxx) + 1e-10)
        feats[f"dom_freq_{name}"] = f[np.argmax(pxx)]
        feats[f"spectral_entropy_{name}"] = -np.sum(p * np.log(p + 1e-10))
        feats[f"low_freq_power_{name}"] = _trapz(pxx[(f >= 0.1) & (f <= 2.0)])
        feats[f"high_freq_power_{name}"] = _trapz(pxx[(f >= 2.0) & (f <= 10.0)])
        feats[f"spectral_centroid_{name}"] = (np.sum(f * pxx) / np.sum(pxx)
                                              if np.sum(pxx) > 0 else 0)
    return feats


def compute_hrv_features(ibi_ms):
    if len(ibi_ms) < 5:
        return {}
    ibi = np.array(ibi_ms, dtype=float) / 1000.0
    ibi = ibi[(ibi > 0.3) & (ibi < 2.0)]
    if len(ibi) < 5:
        return {}

    d = np.diff(ibi)
    lf = hf = lfhf = tot = lfn = hfn = 0
    ta = np.cumsum(ibi) - np.cumsum(ibi)[0]
    ut = np.arange(0, ta[-1], 1 / 4.0)
    if len(ut) >= 5:
        interp = np.interp(ut, ta, ibi) - np.mean(np.interp(ut, ta, ibi))
        fq, pxx = welch(interp, fs=4.0, nperseg=min(256, len(interp)))
        lf = _trapz(pxx[(fq >= 0.04) & (fq <= 0.15)])
        hf = _trapz(pxx[(fq >= 0.15) & (fq <= 0.40)])
        tot = _trapz(pxx)
        lfhf = lf / hf if hf > 0 else 0
        lfn = lf / tot if tot > 0 else 0
        hfn = hf / tot if tot > 0 else 0

    return {
        "rmssd": np.sqrt(np.mean(d**2)) if len(d) > 0 else 0,
        "sdnn": np.std(ibi),
        "mean_hr": 60.0 / np.mean(ibi) if np.mean(ibi) > 0 else 0,
        "pnn50": np.sum(np.abs(d) > 0.05) / len(d) if len(d) > 0 else 0,
        "mean_ibi": np.mean(ibi),
        "ibi_skew": float(skew(ibi)) if len(ibi) > 2 else 0,
        "ibi_kurt": float(kurtosis(ibi)) if len(ibi) > 3 else 0,
        "lf_power": lf, "hf_power": hf, "lf_hf_ratio": lfhf,
        "total_power": tot, "lf_norm": lfn, "hf_norm": hfn,
    }


# ── Input handling ───────────────────────────────────────────────────
def files_from_upload(uploads):
    """Map uploaded files (CSVs or one zip) to {basename_lower: bytes}."""
    out = {}
    for up in uploads:
        data = up.getvalue()
        if up.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for name in z.namelist():
                    base = os.path.basename(name)
                    if base.lower().endswith(".csv") and not base.startswith("._"):
                        out[base.lower()] = z.read(name)
        else:
            out[up.name.lower()] = data
    return out


def _find(files, key):
    for name, data in files.items():
        if key in name:
            return pd.read_csv(io.BytesIO(data))
    return None


def read_session(files):
    """Return (acc, gyro, hr) DataFrames; gyro and hr may be None."""
    acc = _find(files, "accelerometer")
    if acc is None:
        raise ValueError("No Accelerometer.csv found. Upload the session folder "
                         "as a zip, or add Accelerometer.csv.")
    if not {"x", "y", "z"}.issubset(acc.columns):
        raise ValueError("Accelerometer.csv needs x, y and z columns.")
    if len(acc) < WINDOW_LEN + STEP_LEN:
        raise ValueError(f"The recording is {len(acc) / FS:.0f} s long. "
                         f"At least {(WINDOW_LEN + STEP_LEN) / FS:.0f} s is needed.")
    return acc, _find(files, "gyroscope"), _find(files, "heartrate")


# ── Windowing and features ───────────────────────────────────────────
def build_features(acc, gyr, hr):
    ax = acc["x"].values.astype(float)
    ay = acc["y"].values.astype(float)
    az = acc["z"].values.astype(float)

    gyro_ok = gyr is not None and len(gyr) >= len(ax) - 10
    if gyro_ok:
        gx = gyr["x"].values[:len(ax)].astype(float)
        gy = gyr["y"].values[:len(ax)].astype(float)
        gz = gyr["z"].values[:len(ax)].astype(float)
    else:
        gx = gy = gz = np.zeros_like(ax)
    n = min(len(ax), len(gx))

    ibi_ms = []
    if hr is not None and "ibi_ms" in hr.columns:
        ibi_ms = hr["ibi_ms"].dropna().values.tolist()
    # HRV is computed once over the whole session, as in training.
    hrv = compute_hrv_features(ibi_ms)

    rows, starts = [], []
    t0 = time.perf_counter()
    for s in range(0, n - WINDOW_LEN, STEP_LEN):
        e = s + WINDOW_LEN
        row = compute_imu_features(ax[s:e], ay[s:e], az[s:e], gx[s:e], gy[s:e], gz[s:e])
        row.update(hrv)
        vals = np.array(list(row.values()), dtype=float)
        if not np.all(np.isfinite(vals)):
            continue
        rows.append(row)
        starts.append(s / FS / 60.0)
    feat_ms = (time.perf_counter() - t0) * 1000 / max(len(rows), 1)

    info = {
        "duration_min": n / FS / 60.0,
        "gyro_used": gyro_ok,
        "ibi_count": len(ibi_ms),
        "hrv": hrv,
        "feature_ms_per_window": feat_ms,
    }
    return pd.DataFrame(rows).fillna(0), np.array(starts), info


# ── Models and inference ─────────────────────────────────────────────
def load_models(model_dir):
    model_dir = os.path.expanduser(model_dir)
    missing = [f for f in MODEL_FILES.values()
               if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        raise FileNotFoundError(f"Missing in {model_dir}: {', '.join(missing)}")
    m = {k: joblib.load(os.path.join(model_dir, f)) for k, f in MODEL_FILES.items()}
    if hasattr(m["rf"], "n_jobs"):
        m["rf"].n_jobs = 1
    try:
        m["xgb"].set_params(n_jobs=1)
    except Exception:
        pass
    m["size_mb"] = sum(os.path.getsize(os.path.join(model_dir, f))
                       for f in MODEL_FILES.values()) / 1e6
    return m


def predict(models, feats):
    X = feats.reindex(columns=list(models["features"]), fill_value=0).values
    Xs = models["scaler"].transform(X)
    per_model = {k: models[k].predict_proba(Xs)[:, 1] for k in ("rf", "xgb", "gb")}
    proba = sum(WEIGHTS[k] * per_model[k] for k in WEIGHTS)

    one = Xs[:1]
    runs = []
    for _ in range(20):
        t = time.perf_counter()
        _ = sum(WEIGHTS[k] * models[k].predict_proba(one)[:, 1] for k in WEIGHTS)
        runs.append((time.perf_counter() - t) * 1000)

    score = float(np.mean(proba))
    threshold = float(models["threshold"])
    return {
        "proba": proba,
        "per_model": per_model,
        "score": score,
        "threshold": threshold,
        "fatigued": score >= threshold,
        "infer_ms_per_window": float(np.median(runs)),
    }


# ── Synthetic demo session (for visitors without watch data) ─────────
def make_demo_session(minutes=20, seed=7):
    """Synthetic signals with the watch app's file layout.

    Motion restlessness and heart rate rise slowly while HRV falls, so
    the demo shows how the timeline evolves. It is not real data.
    """
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * FS)
    t = np.arange(n) / FS
    drift = np.linspace(0, 1, n)
    fidget = (0.02 + 0.05 * drift) * rng.standard_normal(n)
    acc = pd.DataFrame({
        "timestamp_sec": t,
        "x": 0.05 + fidget,
        "y": -0.42 + 0.8 * fidget,
        "z": -0.90 + 0.6 * fidget,
    })
    gyr = pd.DataFrame({
        "timestamp_sec": t,
        "x": (0.10 + 0.25 * drift) * rng.standard_normal(n),
        "y": (0.10 + 0.25 * drift) * rng.standard_normal(n),
        "z": (0.10 + 0.25 * drift) * rng.standard_normal(n),
    })
    beats, ibis, clock = [], [], 0.0
    while clock < minutes * 60:
        frac = clock / (minutes * 60)
        ibi = 760 - 90 * frac + (40 - 25 * frac) * rng.standard_normal()
        clock += ibi / 1000
        beats.append(clock)
        ibis.append(ibi)
    beats, ibis = np.array(beats), np.array(ibis)
    keep = rng.random(len(ibis)) < 0.2  # the watch reports only some beats
    hr = pd.DataFrame({
        "timestamp_unix_ms": 1.7e12 + beats[keep] * 1000,
        "bpm": 60000 / ibis[keep],
        "ibi_ms": ibis[keep],
    })
    return acc, gyr, hr
