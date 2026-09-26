import os
from pathlib import Path
from string import Template

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from pipeline import (WEIGHTS, build_features, files_from_upload, load_models,
                      make_demo_session, predict, read_session)

st.set_page_config(page_title="EdgeFatigue", page_icon="⌚", layout="centered",
                   initial_sidebar_state="collapsed")

C = dict(INDIGO="#2D3A8C", INK="#1B2340", MUTED="#5A6482", LINE="#DDE3F0",
         LOW="#1F9D74", LOW_T="#E1F4EC", MID="#E0A020", MID_T="#FCF1D8",
         HIGH="#D6453D", HIGH_T="#FBE4E2", ROSE="#D9467A", VIOLET="#6C5CE7")

st.markdown(Template("""<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, button, input { font-family: 'Manrope', system-ui, sans-serif !important; }
.block-container { max-width: 860px; padding-top: 3.6rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
h3 { font-weight: 800 !important; color: $INK; letter-spacing: -0.01em; margin-top: 2rem !important; }
.hero { background: $INDIGO; border-radius: 18px; padding: 1.8rem 2rem 1.4rem; margin-bottom: 1.2rem; }
.hero h1 { color: #fff !important; font-size: 2.1rem !important; font-weight: 800 !important;
  line-height: 1.15 !important; letter-spacing: -0.02em; margin: 0 0 0.5rem; padding: 0 !important; }
.hero p { color: #D9DEF5; font-size: 1.05rem; margin: 0 0 1.1rem; max-width: 58ch; }
.scale { display: flex; height: 8px; border-radius: 4px; overflow: hidden; max-width: 330px; }
.scale span { flex: 1; }
.scale-labels { display: flex; justify-content: space-between; max-width: 330px;
  font-size: 0.8rem; color: #D9DEF5; margin-top: 0.35rem; }
[data-testid="stFileUploaderDropzone"] { background: #fff; border: 2px dashed $INDIGO; border-radius: 14px; }
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.9rem; margin: 1.3rem 0; }
.step { background: #fff; border-radius: 14px; padding: 1rem 1.1rem; border-top: 4px solid var(--c); }
.step .n { display: inline-grid; place-items: center; width: 1.8rem; height: 1.8rem; border-radius: 50%;
  background: var(--c); color: #fff; font-weight: 800; font-size: 0.9rem; }
.step .t { font-weight: 800; color: $INK; margin: 0.55rem 0 0.2rem; }
.step .d { color: $MUTED; font-size: 0.92rem; line-height: 1.5; }
.verdict { display: flex; align-items: center; gap: 1.6rem; border-radius: 18px;
  padding: 1.4rem 1.6rem; margin: 0.6rem 0 0.4rem; }
.ring { flex: none; width: 140px; height: 140px; border-radius: 50%; display: grid; place-items: center; }
.ring > div { width: 108px; height: 108px; border-radius: 50%; background: #fff; display: flex;
  flex-direction: column; align-items: center; justify-content: center; }
.ring b { font-size: 2.5rem; font-weight: 800; line-height: 1; }
.ring small { color: $MUTED; font-size: 0.8rem; }
.verdict .lv { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.05; }
.verdict .txt { color: $INK; font-size: 1.02rem; margin-top: 0.45rem; max-width: 48ch; line-height: 1.55; }
.strip { display: flex; height: 44px; border-radius: 8px; overflow: hidden; margin-top: 0.5rem; }
.strip span { flex: 1 1 0; min-width: 0; }
.ticks { position: relative; height: 1.3rem; font-size: 0.8rem; color: $MUTED; }
.ticks span { position: absolute; top: 0.25rem; transform: translateX(-50%); }
.ticks span:first-child { transform: none; }
.phases, .cards { display: grid; gap: 0.9rem; }
.phases { grid-template-columns: repeat(3, 1fr); }
.cards { grid-template-columns: repeat(2, 1fr); }
.phase { border-radius: 14px; padding: 0.9rem 1.1rem; }
.phase .name { font-size: 0.88rem; color: $MUTED; font-weight: 600; }
.phase .val { font-size: 2.3rem; font-weight: 800; line-height: 1.1; }
.phase .lvl { font-size: 0.88rem; font-weight: 700; }
.card { background: #fff; border-radius: 14px; padding: 1rem 1.1rem; border-left: 5px solid var(--c); }
.card .h { font-weight: 800; color: $INK; }
.card .d { color: $MUTED; font-size: 0.93rem; margin-top: 0.25rem; line-height: 1.5; }
.note { font-size: 0.85rem; color: $MUTED; }
.foot { margin-top: 2.2rem; font-size: 0.82rem; color: $MUTED; }
@media (max-width: 640px) {
  .steps, .phases, .cards { grid-template-columns: 1fr; }
  .verdict { flex-direction: column; align-items: flex-start; } }
</style>""").substitute(C), unsafe_allow_html=True)

# ── Models ───────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent
_default = os.environ.get("EDGEFATIGUE_MODELS", str(APP_DIR / "models"))
if not any(Path(_default).expanduser().glob("*.pkl")) and \
        Path("~/fatigue_26/watch_models").expanduser().exists():
    _default = "~/fatigue_26/watch_models"
st.session_state.setdefault("model_dir", _default)


@st.cache_resource(show_spinner="Loading models")
def get_models(path):
    return load_models(path)


@st.cache_data(show_spinner=False)
def analyse(acc, gyr, hr, model_path):
    feats, starts, info = build_features(acc, gyr, hr)
    if feats.empty:
        return None
    return feats, starts, info, predict(get_models(model_path), feats)


def level_of(p, thr):
    if p >= thr:
        return "Fatigued", C["HIGH"], C["HIGH_T"]
    if p >= thr - 0.10:
        return "Borderline", C["MID"], C["MID_T"]
    return "Low fatigue", C["LOW"], C["LOW_T"]


def footer_and_settings():
    with st.expander("Model settings"):
        st.text_input("Model folder", key="model_dir")
        st.caption("Trained on 16 people; 75% accurate on people it hasn't seen.")
    st.markdown("<div class='foot'>Research prototype. Not a medical device.</div>",
                unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────
st.markdown(Template("""<div class='hero'>
<h1>See when fatigue set in during desk work</h1>
<p>Upload an Apple Watch session and get a fatigue score, a minute-by-minute
view, and what changed in movement and heart rate.</p>
<div class='scale'><span style='background:$LOW'></span><span style='background:$MID'></span>
<span style='background:$HIGH'></span></div>
<div class='scale-labels'><span>Low</span><span>Borderline</span><span>Fatigued</span></div>
</div>""").substitute(C), unsafe_allow_html=True)

try:
    models = get_models(st.session_state.model_dir)
except Exception as err:
    st.error(f"Couldn't load the models. {err}")
    st.text_input("Model folder", key="model_dir")
    st.stop()

uploads = st.file_uploader("Upload a session", type=["zip", "csv"],
                           accept_multiple_files=True, label_visibility="collapsed")
if uploads:
    st.session_state.demo = False

if not uploads and not st.session_state.get("demo"):
    st.markdown(Template("""<div class='steps'>
<div class='step' style='--c:$INDIGO'><span class='n'>1</span><div class='t'>Record</div>
<div class='d'>Wear the watch and run the EduFatigue app during desk work.</div></div>
<div class='step' style='--c:$LOW'><span class='n'>2</span><div class='t'>Upload</div>
<div class='d'>Zip the session folder, or select its CSV files above.</div></div>
<div class='step' style='--c:$MID'><span class='n'>3</span><div class='t'>See results</div>
<div class='d'>Get a fatigue score and when it changed during the session.</div></div>
</div>""").substitute(C), unsafe_allow_html=True)
    with st.expander("Which files do I need?"):
        st.markdown("**Accelerometer.csv** is required. **Gyroscope.csv** and "
                    "**HeartRate_IBI.csv** make the estimate better.")
    if st.button("Try a demo session", type="primary"):
        st.session_state.demo = True
        st.rerun()
    footer_and_settings()
    st.stop()

if st.session_state.get("demo") and not uploads:
    acc, gyr, hr = make_demo_session()
    st.info("Demo: a synthetic 20-minute session. Its result says nothing about a real person.")
else:
    try:
        acc, gyr, hr = read_session(files_from_upload(uploads))
    except Exception as err:
        st.error(str(err))
        st.stop()

with st.spinner("Reading the session"):
    out = analyse(acc, gyr, hr, st.session_state.model_dir)
if out is None:
    st.error("This recording has no usable 20-second stretches. "
             "Check that Accelerometer.csv covers at least 30 seconds.")
    st.stop()
feats, starts, info, res = out
proba, thr, score = res["proba"], res["threshold"], res["score"]
level, color, tint = level_of(score, thr)

# ── Insights ─────────────────────────────────────────────────────────
above = proba >= thr
onset, run = None, 0
for i, a in enumerate(above):
    run = run + 1 if a else 0
    if run >= 3:
        onset = starts[i - 2]
        break
smooth = pd.Series(proba).rolling(6, min_periods=1, center=True).mean().values
peak_min = float(starts[int(np.argmax(smooth))])
thirds = np.array_split(np.arange(len(proba)), 3)
phase_scores = [float(proba[idx].mean()) for idx in thirds]
phase_edges = [(starts[idx[0]], starts[idx[-1]] + 20 / 60) for idx in thirds]

restless = None
if "acc_mag_std" in feats.columns and len(feats) >= 6:
    m = feats["acc_mag_std"].values
    a0, a1 = m[thirds[0]].mean(), m[thirds[2]].mean()
    restless = (a1 - a0) / a0 * 100 if a0 > 0 else None
hr_change = None
if hr is not None and {"timestamp_unix_ms", "bpm"}.issubset(hr.columns) and len(hr) > 10:
    h = hr.dropna(subset=["bpm"])
    k = max(len(h) // 3, 1)
    hr_change = h["bpm"].iloc[-k:].mean() - h["bpm"].iloc[:k].mean()

# ── Verdict ──────────────────────────────────────────────────────────
share = above.mean() * 100
when = (f"sustained from minute {onset:.0f}" if onset is not None
        else "never for a sustained stretch")
st.markdown(f"""<div class='verdict' style='background:{tint}'>
<div class='ring' style='background:conic-gradient({color} {score * 100:.0f}%, #ffffff 0)'>
<div><b style='color:{color}'>{score * 100:.0f}</b><small>of 100</small></div></div>
<div><div class='lv' style='color:{color}'>{level}</div>
<div class='txt'>The model saw fatigue in {share:.0f}% of this session, {when}.</div></div>
</div>""", unsafe_allow_html=True)


def mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def shade(p):
    if p < thr:
        return mix(C["LOW"], C["MID"], p / thr)
    return mix(C["MID"], C["HIGH"], (p - thr) / (1 - thr))


st.markdown("### Across the session")
dur = float(starts[-1] + 20 / 60)
strip = "".join(f"<span style='background:{shade(p)}' title='Minute {m:.1f}: "
                f"score {p * 100:.0f}'></span>" for m, p in zip(starts, proba))
every = 5 if dur > 15 else 2
ticks = "".join(f"<span style='left:{m / dur * 100:.2f}%'>{m} min</span>"
                for m in range(0, int(dur) + 1, every))
st.markdown(f"<div class='strip'>{strip}</div><div class='ticks'>{ticks}</div>"
            "<p class='note' style='margin-top:0.3rem'>Each block is 20 seconds. "
            "Hover one to see its score.</p>", unsafe_allow_html=True)

st.markdown("### How it changed")
tiles = ""
for name, (a, b), p in zip(["Start", "Middle", "End"], phase_edges, phase_scores):
    lv, c, t = level_of(p, thr)
    tiles += (f"<div class='phase' style='background:{t}'>"
              f"<div class='name'>{name}, minutes {a:.0f} to {b:.0f}</div>"
              f"<div class='val' style='color:{c}'>{p * 100:.0f}</div>"
              f"<div class='lvl' style='color:{c}'>{lv}</div></div>")
st.markdown(f"<div class='phases'>{tiles}</div>", unsafe_allow_html=True)

cards = []
cards.append((C["MID"], "⏱️ Fatigue set in",
              f"Around minute {onset:.0f}. A short break a few minutes earlier might have helped."
              if onset is not None else "Not at any sustained point in this session."))
cards.append((C["HIGH"], "📈 Peak", f"Highest around minute {peak_min:.0f}."))
if restless is not None:
    cards.append((C["VIOLET"], "🖐️ Movement",
                  f"Wrist movement was {abs(restless):.0f}% "
                  f"{'more' if restless > 0 else 'less'} restless at the end than at the start."))
if hr_change is not None:
    cards.append((C["ROSE"], "❤️ Heart rate",
                  f"{abs(hr_change):.0f} bpm {'higher' if hr_change > 0 else 'lower'} "
                  f"at the end than at the start."))
st.markdown("### What stood out")
st.markdown("<div class='cards'>" + "".join(
    f"<div class='card' style='--c:{c}'><div class='h'>{h}</div><div class='d'>{d}</div></div>"
    for c, h, d in cards) + "</div>", unsafe_allow_html=True)

if not info["gyro_used"]:
    st.warning("Gyroscope data was missing or too short, so its features were set to zero, as in training.")
if not info["hrv"]:
    st.warning("Fewer than five heartbeats were recorded, so heart-rate variability features were set to zero.")

# ── Details ──────────────────────────────────────────────────────────
st.markdown("### Details")
t1, t2, t3, t4 = st.tabs(["Score over time", "Heart rate", "Model", "Footprint"])
with t1:
    df = pd.DataFrame({"Minute": starts, "Score": proba * 100, "Smoothed": smooth * 100})
    raw = alt.Chart(df).mark_line(color=C["LINE"], strokeWidth=1).encode(
        x=alt.X("Minute:Q", title="Minute of session"),
        y=alt.Y("Score:Q", title="Fatigue score", scale=alt.Scale(domain=[0, 100])))
    sm = alt.Chart(df).mark_line(color=C["INDIGO"], strokeWidth=3).encode(
        x="Minute:Q", y="Smoothed:Q",
        tooltip=[alt.Tooltip("Minute:Q", format=".1f"),
                 alt.Tooltip("Smoothed:Q", title="Score", format=".0f")])
    rule = alt.Chart(pd.DataFrame({"t": [thr * 100]})).mark_rule(
        color=C["HIGH"], strokeDash=[5, 4]).encode(y="t:Q")
    st.altair_chart((raw + sm + rule).configure_view(stroke=None), width="stretch")
    st.markdown(f"<p class='note'>Thin line: each window. Thick line: one-minute average. "
                f"Red dashes: the fatigue threshold ({thr * 100:.0f}). Borderline means "
                f"within 10 points below it. The model was trained to separate fatigued "
                f"from not fatigued; the levels are ranges of its score.</p>",
                unsafe_allow_html=True)
with t2:
    hrv = info["hrv"]
    if hr is not None and {"timestamp_unix_ms", "bpm"}.issubset(hr.columns) and len(hr) > 1:
        h = hr.dropna(subset=["bpm"]).copy()
        h["Minute"] = (h["timestamp_unix_ms"] - h["timestamp_unix_ms"].iloc[0]) / 60000
        st.altair_chart(alt.Chart(h).mark_line(color=C["ROSE"], strokeWidth=2).encode(
            x=alt.X("Minute:Q", title="Minute of session"),
            y=alt.Y("bpm:Q", title="Heart rate (bpm)", scale=alt.Scale(zero=False))
        ).configure_view(stroke=None), width="stretch")
    else:
        st.write("No heart-rate file was provided.")
    if hrv:
        a, b, c, d = st.columns(4)
        a.metric("Mean HR", f"{hrv['mean_hr']:.0f} bpm")
        b.metric("RMSSD", f"{hrv['rmssd'] * 1000:.0f} ms")
        c.metric("SDNN", f"{hrv['sdnn'] * 1000:.0f} ms")
        d.metric("LF/HF", f"{hrv['lf_hf_ratio']:.2f}")
with t3:
    names = {"rf": "Random Forest", "xgb": "XGBoost", "gb": "Gradient Boosting"}
    agree = pd.DataFrame({"Model": [names[k] for k in WEIGHTS],
                          "Score": [float(np.mean(res["per_model"][k])) * 100 for k in WEIGHTS]})
    bars = alt.Chart(agree).mark_bar(height=20, cornerRadiusEnd=4).encode(
        x=alt.X("Score:Q", title="Mean fatigue score", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Model:N", title=None, sort=None),
        color=alt.Color("Model:N", legend=None,
                        scale=alt.Scale(range=[C["INDIGO"], C["VIOLET"], C["LOW"]])),
        tooltip=["Model", alt.Tooltip("Score:Q", format=".0f")])
    vr = alt.Chart(pd.DataFrame({"t": [thr * 100]})).mark_rule(
        color=C["HIGH"], strokeDash=[5, 4]).encode(x="t:Q")
    st.altair_chart((bars + vr).properties(height=130).configure_view(stroke=None),
                    width="stretch")
    st.markdown("<p class='note'>The three models are combined with weights 0.35, 0.40 "
                "and 0.25.</p>", unsafe_allow_html=True)
with t4:
    a, b, c = st.columns(3)
    a.metric("Model size", f"{models['size_mb']:.1f} MB")
    b.metric("Features per window", f"{info['feature_ms_per_window']:.1f} ms")
    c.metric("Inference per window", f"{res['infer_ms_per_window']:.1f} ms")
    st.markdown("<p class='note'>Measured on the machine running this app, not on the watch.</p>",
                unsafe_allow_html=True)

table = pd.DataFrame({
    "start_min": np.round(starts, 2), "fatigue_score": np.round(proba * 100, 1),
    "random_forest": np.round(res["per_model"]["rf"] * 100, 1),
    "xgboost": np.round(res["per_model"]["xgb"] * 100, 1),
    "gradient_boosting": np.round(res["per_model"]["gb"] * 100, 1)})
st.download_button("Download scores as CSV", table.to_csv(index=False),
                   file_name="edgefatigue_scores.csv", mime="text/csv")
footer_and_settings()
