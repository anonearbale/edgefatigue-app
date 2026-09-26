# import os
# from pathlib import Path
# from string import Template

# import altair as alt
# import numpy as np
# import pandas as pd
# import streamlit as st

# from pipeline import (WEIGHTS, build_features, files_from_upload, load_models,
#                       make_demo_session, predict, read_session)

# st.set_page_config(page_title="EdgeFatigue", page_icon="⌚", layout="centered",
#                    initial_sidebar_state="collapsed")

# C = dict(INDIGO="#2D3A8C", INK="#1B2340", MUTED="#5A6482", LINE="#DDE3F0",
#          LOW="#1F9D74", LOW_T="#E1F4EC", MID="#E0A020", MID_T="#FCF1D8",
#          HIGH="#D6453D", HIGH_T="#FBE4E2", ROSE="#D9467A", VIOLET="#6C5CE7")

# st.markdown(Template("""<style>
# @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
# html, body, .stApp, .stMarkdown, button, input { font-family: 'Manrope', system-ui, sans-serif !important; }
# .block-container { max-width: 860px; padding-top: 3.6rem; padding-bottom: 3rem; }
# header[data-testid="stHeader"] { background: transparent; }
# [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
# h3 { font-weight: 800 !important; color: $INK; letter-spacing: -0.01em; margin-top: 2rem !important; }
# .hero { background: $INDIGO; border-radius: 18px; padding: 1.8rem 2rem 1.4rem; margin-bottom: 1.2rem; }
# .hero h1 { color: #fff !important; font-size: 2.1rem !important; font-weight: 800 !important;
#   line-height: 1.15 !important; letter-spacing: -0.02em; margin: 0 0 0.5rem; padding: 0 !important; }
# .hero p { color: #D9DEF5; font-size: 1.05rem; margin: 0 0 1.1rem; max-width: 58ch; }
# .scale { display: flex; height: 8px; border-radius: 4px; overflow: hidden; max-width: 330px; }
# .scale span { flex: 1; }
# .scale-labels { display: flex; justify-content: space-between; max-width: 330px;
#   font-size: 0.8rem; color: #D9DEF5; margin-top: 0.35rem; }
# [data-testid="stFileUploaderDropzone"] { background: #fff; border: 2px dashed $INDIGO; border-radius: 14px; }
# .steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.9rem; margin: 1.3rem 0; }
# .step { background: #fff; border-radius: 14px; padding: 1rem 1.1rem; border-top: 4px solid var(--c); }
# .step .n { display: inline-grid; place-items: center; width: 1.8rem; height: 1.8rem; border-radius: 50%;
#   background: var(--c); color: #fff; font-weight: 800; font-size: 0.9rem; }
# .step .t { font-weight: 800; color: $INK; margin: 0.55rem 0 0.2rem; }
# .step .d { color: $MUTED; font-size: 0.92rem; line-height: 1.5; }
# .verdict { display: flex; align-items: center; gap: 1.6rem; border-radius: 18px;
#   padding: 1.4rem 1.6rem; margin: 0.6rem 0 0.4rem; }
# .ring { flex: none; width: 140px; height: 140px; border-radius: 50%; display: grid; place-items: center; }
# .ring > div { width: 108px; height: 108px; border-radius: 50%; background: #fff; display: flex;
#   flex-direction: column; align-items: center; justify-content: center; }
# .ring b { font-size: 2.5rem; font-weight: 800; line-height: 1; }
# .ring small { color: $MUTED; font-size: 0.8rem; }
# .verdict .lv { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.05; }
# .verdict .txt { color: $INK; font-size: 1.02rem; margin-top: 0.45rem; max-width: 48ch; line-height: 1.55; }
# .strip { display: flex; height: 44px; border-radius: 8px; overflow: hidden; margin-top: 0.5rem; }
# .strip span { flex: 1 1 0; min-width: 0; }
# .ticks { position: relative; height: 1.3rem; font-size: 0.8rem; color: $MUTED; }
# .ticks span { position: absolute; top: 0.25rem; transform: translateX(-50%); }
# .ticks span:first-child { transform: none; }
# .phases, .cards { display: grid; gap: 0.9rem; }
# .phases { grid-template-columns: repeat(3, 1fr); }
# .cards { grid-template-columns: repeat(2, 1fr); }
# .phase { border-radius: 14px; padding: 0.9rem 1.1rem; }
# .phase .name { font-size: 0.88rem; color: $MUTED; font-weight: 600; }
# .phase .val { font-size: 2.3rem; font-weight: 800; line-height: 1.1; }
# .phase .lvl { font-size: 0.88rem; font-weight: 700; }
# .card { background: #fff; border-radius: 14px; padding: 1rem 1.1rem; border-left: 5px solid var(--c); }
# .card .h { font-weight: 800; color: $INK; }
# .card .d { color: $MUTED; font-size: 0.93rem; margin-top: 0.25rem; line-height: 1.5; }
# .note { font-size: 0.85rem; color: $MUTED; }
# .foot { margin-top: 2.2rem; font-size: 0.82rem; color: $MUTED; }
# @media (max-width: 640px) {
#   .steps, .phases, .cards { grid-template-columns: 1fr; }
#   .verdict { flex-direction: column; align-items: flex-start; } }
# </style>""").substitute(C), unsafe_allow_html=True)

# # ── Models ───────────────────────────────────────────────────────────
# APP_DIR = Path(__file__).parent
# _default = os.environ.get("EDGEFATIGUE_MODELS", str(APP_DIR / "models"))
# if not any(Path(_default).expanduser().glob("*.pkl")) and \
#         Path("~/fatigue_26/watch_models").expanduser().exists():
#     _default = "~/fatigue_26/watch_models"
# st.session_state.setdefault("model_dir", _default)


# @st.cache_resource(show_spinner="Loading models")
# def get_models(path):
#     return load_models(path)


# @st.cache_data(show_spinner=False)
# def analyse(acc, gyr, hr, model_path):
#     feats, starts, info = build_features(acc, gyr, hr)
#     if feats.empty:
#         return None
#     return feats, starts, info, predict(get_models(model_path), feats)


# def level_of(p, thr):
#     if p >= thr:
#         return "Fatigued", C["HIGH"], C["HIGH_T"]
#     if p >= thr - 0.10:
#         return "Borderline", C["MID"], C["MID_T"]
#     return "Low fatigue", C["LOW"], C["LOW_T"]


# def footer_and_settings():
#     with st.expander("Model settings"):
#         st.text_input("Model folder", key="model_dir")
#         st.caption("Trained on 16 people; 75% accurate on people it hasn't seen.")
#     st.markdown("<div class='foot'>Research prototype. Not a medical device.</div>",
#                 unsafe_allow_html=True)


# # ── Header ───────────────────────────────────────────────────────────
# st.markdown(Template("""<div class='hero'>
# <h1>See when fatigue set in during desk work</h1>
# <p>Upload an Apple Watch session and get a fatigue score, a minute-by-minute
# view, and what changed in movement and heart rate.</p>
# <div class='scale'><span style='background:$LOW'></span><span style='background:$MID'></span>
# <span style='background:$HIGH'></span></div>
# <div class='scale-labels'><span>Low</span><span>Borderline</span><span>Fatigued</span></div>
# </div>""").substitute(C), unsafe_allow_html=True)

# try:
#     models = get_models(st.session_state.model_dir)
# except Exception as err:
#     st.error(f"Couldn't load the models. {err}")
#     st.text_input("Model folder", key="model_dir")
#     st.stop()

# uploads = st.file_uploader("Upload a session", type=["zip", "csv"],
#                            accept_multiple_files=True, label_visibility="collapsed")
# if uploads:
#     st.session_state.demo = False

# if not uploads and not st.session_state.get("demo"):
#     st.markdown(Template("""<div class='steps'>
# <div class='step' style='--c:$INDIGO'><span class='n'>1</span><div class='t'>Record</div>
# <div class='d'>Wear the watch and run the EduFatigue app during desk work.</div></div>
# <div class='step' style='--c:$LOW'><span class='n'>2</span><div class='t'>Upload</div>
# <div class='d'>Zip the session folder, or select its CSV files above.</div></div>
# <div class='step' style='--c:$MID'><span class='n'>3</span><div class='t'>See results</div>
# <div class='d'>Get a fatigue score and when it changed during the session.</div></div>
# </div>""").substitute(C), unsafe_allow_html=True)
#     with st.expander("Which files do I need?"):
#         st.markdown("**Accelerometer.csv** is required. **Gyroscope.csv** and "
#                     "**HeartRate_IBI.csv** make the estimate better.")
#     if st.button("Try a demo session", type="primary"):
#         st.session_state.demo = True
#         st.rerun()
#     footer_and_settings()
#     st.stop()

# if st.session_state.get("demo") and not uploads:
#     acc, gyr, hr = make_demo_session()
#     st.info("Demo: a synthetic 20-minute session. Its result says nothing about a real person.")
# else:
#     try:
#         acc, gyr, hr = read_session(files_from_upload(uploads))
#     except Exception as err:
#         st.error(str(err))
#         st.stop()

# with st.spinner("Reading the session"):
#     out = analyse(acc, gyr, hr, st.session_state.model_dir)
# if out is None:
#     st.error("This recording has no usable 20-second stretches. "
#              "Check that Accelerometer.csv covers at least 30 seconds.")
#     st.stop()
# feats, starts, info, res = out
# proba, thr, score = res["proba"], res["threshold"], res["score"]
# level, color, tint = level_of(score, thr)

# # ── Insights ─────────────────────────────────────────────────────────
# above = proba >= thr
# onset, run = None, 0
# for i, a in enumerate(above):
#     run = run + 1 if a else 0
#     if run >= 3:
#         onset = starts[i - 2]
#         break
# smooth = pd.Series(proba).rolling(6, min_periods=1, center=True).mean().values
# peak_min = float(starts[int(np.argmax(smooth))])
# thirds = np.array_split(np.arange(len(proba)), 3)
# phase_scores = [float(proba[idx].mean()) for idx in thirds]
# phase_edges = [(starts[idx[0]], starts[idx[-1]] + 20 / 60) for idx in thirds]

# restless = None
# if "acc_mag_std" in feats.columns and len(feats) >= 6:
#     m = feats["acc_mag_std"].values
#     a0, a1 = m[thirds[0]].mean(), m[thirds[2]].mean()
#     restless = (a1 - a0) / a0 * 100 if a0 > 0 else None
# hr_change = None
# if hr is not None and {"timestamp_unix_ms", "bpm"}.issubset(hr.columns) and len(hr) > 10:
#     h = hr.dropna(subset=["bpm"])
#     k = max(len(h) // 3, 1)
#     hr_change = h["bpm"].iloc[-k:].mean() - h["bpm"].iloc[:k].mean()

# # ── Verdict ──────────────────────────────────────────────────────────
# share = above.mean() * 100
# when = (f"sustained from minute {onset:.0f}" if onset is not None
#         else "never for a sustained stretch")
# st.markdown(f"""<div class='verdict' style='background:{tint}'>
# <div class='ring' style='background:conic-gradient({color} {score * 100:.0f}%, #ffffff 0)'>
# <div><b style='color:{color}'>{score * 100:.0f}</b><small>of 100</small></div></div>
# <div><div class='lv' style='color:{color}'>{level}</div>
# <div class='txt'>The model saw fatigue in {share:.0f}% of this session, {when}.</div></div>
# </div>""", unsafe_allow_html=True)


# def mix(c1, c2, t):
#     a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
#     b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
#     return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


# def shade(p):
#     if p < thr:
#         return mix(C["LOW"], C["MID"], p / thr)
#     return mix(C["MID"], C["HIGH"], (p - thr) / (1 - thr))


# st.markdown("### Across the session")
# dur = float(starts[-1] + 20 / 60)
# strip = "".join(f"<span style='background:{shade(p)}' title='Minute {m:.1f}: "
#                 f"score {p * 100:.0f}'></span>" for m, p in zip(starts, proba))
# every = 5 if dur > 15 else 2
# ticks = "".join(f"<span style='left:{m / dur * 100:.2f}%'>{m} min</span>"
#                 for m in range(0, int(dur) + 1, every))
# st.markdown(f"<div class='strip'>{strip}</div><div class='ticks'>{ticks}</div>"
#             "<p class='note' style='margin-top:0.3rem'>Each block is 20 seconds. "
#             "Hover one to see its score.</p>", unsafe_allow_html=True)

# st.markdown("### How it changed")
# tiles = ""
# for name, (a, b), p in zip(["Start", "Middle", "End"], phase_edges, phase_scores):
#     lv, c, t = level_of(p, thr)
#     tiles += (f"<div class='phase' style='background:{t}'>"
#               f"<div class='name'>{name}, minutes {a:.0f} to {b:.0f}</div>"
#               f"<div class='val' style='color:{c}'>{p * 100:.0f}</div>"
#               f"<div class='lvl' style='color:{c}'>{lv}</div></div>")
# st.markdown(f"<div class='phases'>{tiles}</div>", unsafe_allow_html=True)

# cards = []
# cards.append((C["MID"], "⏱️ Fatigue set in",
#               f"Around minute {onset:.0f}. A short break a few minutes earlier might have helped."
#               if onset is not None else "Not at any sustained point in this session."))
# cards.append((C["HIGH"], "📈 Peak", f"Highest around minute {peak_min:.0f}."))
# if restless is not None:
#     cards.append((C["VIOLET"], "🖐️ Movement",
#                   f"Wrist movement was {abs(restless):.0f}% "
#                   f"{'more' if restless > 0 else 'less'} restless at the end than at the start."))
# if hr_change is not None:
#     cards.append((C["ROSE"], "❤️ Heart rate",
#                   f"{abs(hr_change):.0f} bpm {'higher' if hr_change > 0 else 'lower'} "
#                   f"at the end than at the start."))
# st.markdown("### What stood out")
# st.markdown("<div class='cards'>" + "".join(
#     f"<div class='card' style='--c:{c}'><div class='h'>{h}</div><div class='d'>{d}</div></div>"
#     for c, h, d in cards) + "</div>", unsafe_allow_html=True)

# if not info["gyro_used"]:
#     st.warning("Gyroscope data was missing or too short, so its features were set to zero, as in training.")
# if not info["hrv"]:
#     st.warning("Fewer than five heartbeats were recorded, so heart-rate variability features were set to zero.")

# # ── Details ──────────────────────────────────────────────────────────
# st.markdown("### Details")
# t1, t2, t3, t4 = st.tabs(["Score over time", "Heart rate", "Model", "Footprint"])
# with t1:
#     df = pd.DataFrame({"Minute": starts, "Score": proba * 100, "Smoothed": smooth * 100})
#     raw = alt.Chart(df).mark_line(color=C["LINE"], strokeWidth=1).encode(
#         x=alt.X("Minute:Q", title="Minute of session"),
#         y=alt.Y("Score:Q", title="Fatigue score", scale=alt.Scale(domain=[0, 100])))
#     sm = alt.Chart(df).mark_line(color=C["INDIGO"], strokeWidth=3).encode(
#         x="Minute:Q", y="Smoothed:Q",
#         tooltip=[alt.Tooltip("Minute:Q", format=".1f"),
#                  alt.Tooltip("Smoothed:Q", title="Score", format=".0f")])
#     rule = alt.Chart(pd.DataFrame({"t": [thr * 100]})).mark_rule(
#         color=C["HIGH"], strokeDash=[5, 4]).encode(y="t:Q")
#     st.altair_chart((raw + sm + rule).configure_view(stroke=None), width="stretch")
#     st.markdown(f"<p class='note'>Thin line: each window. Thick line: one-minute average. "
#                 f"Red dashes: the fatigue threshold ({thr * 100:.0f}). Borderline means "
#                 f"within 10 points below it. The model was trained to separate fatigued "
#                 f"from not fatigued; the levels are ranges of its score.</p>",
#                 unsafe_allow_html=True)
# with t2:
#     hrv = info["hrv"]
#     if hr is not None and {"timestamp_unix_ms", "bpm"}.issubset(hr.columns) and len(hr) > 1:
#         h = hr.dropna(subset=["bpm"]).copy()
#         h["Minute"] = (h["timestamp_unix_ms"] - h["timestamp_unix_ms"].iloc[0]) / 60000
#         st.altair_chart(alt.Chart(h).mark_line(color=C["ROSE"], strokeWidth=2).encode(
#             x=alt.X("Minute:Q", title="Minute of session"),
#             y=alt.Y("bpm:Q", title="Heart rate (bpm)", scale=alt.Scale(zero=False))
#         ).configure_view(stroke=None), width="stretch")
#     else:
#         st.write("No heart-rate file was provided.")
#     if hrv:
#         a, b, c, d = st.columns(4)
#         a.metric("Mean HR", f"{hrv['mean_hr']:.0f} bpm")
#         b.metric("RMSSD", f"{hrv['rmssd'] * 1000:.0f} ms")
#         c.metric("SDNN", f"{hrv['sdnn'] * 1000:.0f} ms")
#         d.metric("LF/HF", f"{hrv['lf_hf_ratio']:.2f}")
# with t3:
#     names = {"rf": "Random Forest", "xgb": "XGBoost", "gb": "Gradient Boosting"}
#     agree = pd.DataFrame({"Model": [names[k] for k in WEIGHTS],
#                           "Score": [float(np.mean(res["per_model"][k])) * 100 for k in WEIGHTS]})
#     bars = alt.Chart(agree).mark_bar(height=20, cornerRadiusEnd=4).encode(
#         x=alt.X("Score:Q", title="Mean fatigue score", scale=alt.Scale(domain=[0, 100])),
#         y=alt.Y("Model:N", title=None, sort=None),
#         color=alt.Color("Model:N", legend=None,
#                         scale=alt.Scale(range=[C["INDIGO"], C["VIOLET"], C["LOW"]])),
#         tooltip=["Model", alt.Tooltip("Score:Q", format=".0f")])
#     vr = alt.Chart(pd.DataFrame({"t": [thr * 100]})).mark_rule(
#         color=C["HIGH"], strokeDash=[5, 4]).encode(x="t:Q")
#     st.altair_chart((bars + vr).properties(height=130).configure_view(stroke=None),
#                     width="stretch")
#     st.markdown("<p class='note'>The three models are combined with weights 0.35, 0.40 "
#                 "and 0.25.</p>", unsafe_allow_html=True)
# with t4:
#     a, b, c = st.columns(3)
#     a.metric("Model size", f"{models['size_mb']:.1f} MB")
#     b.metric("Features per window", f"{info['feature_ms_per_window']:.1f} ms")
#     c.metric("Inference per window", f"{res['infer_ms_per_window']:.1f} ms")
#     st.markdown("<p class='note'>Measured on the machine running this app, not on the watch.</p>",
#                 unsafe_allow_html=True)

# table = pd.DataFrame({
#     "start_min": np.round(starts, 2), "fatigue_score": np.round(proba * 100, 1),
#     "random_forest": np.round(res["per_model"]["rf"] * 100, 1),
#     "xgboost": np.round(res["per_model"]["xgb"] * 100, 1),
#     "gradient_boosting": np.round(res["per_model"]["gb"] * 100, 1)})
# st.download_button("Download scores as CSV", table.to_csv(index=False),
#                    file_name="edgefatigue_scores.csv", mime="text/csv")
# footer_and_settings()


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

C = dict(INDIGO="#2D3A8C", INK="#1B2340", MUTED="#5A6482", LINE="#C9D1E6",
         LOW="#1F9D74", LOW_T="#E1F4EC", MID="#E0A020", MID_T="#FCF1D8",
         HIGH="#D6453D", HIGH_T="#FBE4E2", ROSE="#D9467A", VIOLET="#6C5CE7")
NAMES = {"rf": "Random Forest", "xgb": "XGBoost", "gb": "Gradient Boosting"}
WIN_MIN = 20 / 60  # window length in minutes

st.markdown(Template("""<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, button, input { font-family: 'Manrope', system-ui, sans-serif !important; }
.block-container { max-width: 900px; padding-top: 3.2rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
h3 { font-weight: 800 !important; color: $INK; letter-spacing: -0.01em; margin-top: 1.6rem !important; }
.hero { background: $INDIGO; border-radius: 18px; padding: 1.5rem 1.8rem 1.2rem; margin-bottom: 1rem; }
.hero h1 { color: #fff !important; font-size: 1.9rem !important; font-weight: 800 !important;
  line-height: 1.15 !important; letter-spacing: -0.02em; margin: 0 0 0.4rem; padding: 0 !important; }
.hero p { color: #D9DEF5; font-size: 1rem; margin: 0; max-width: 60ch; }
[data-testid="stFileUploaderDropzone"] { background: #fff; border: 2px dashed $INDIGO; border-radius: 14px; }
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.9rem; margin: 1.1rem 0; }
.step { background: #fff; border-radius: 14px; padding: 1rem 1.1rem; border-top: 4px solid var(--c); }
.step .n { display: inline-grid; place-items: center; width: 1.8rem; height: 1.8rem; border-radius: 50%;
  background: var(--c); color: #fff; font-weight: 800; font-size: 0.9rem; }
.step .t { font-weight: 800; color: $INK; margin: 0.55rem 0 0.2rem; }
.step .d { color: $MUTED; font-size: 0.92rem; line-height: 1.5; }
.verdict { display: flex; align-items: center; gap: 1.6rem; border-radius: 18px;
  padding: 1.3rem 1.6rem; margin: 0.4rem 0 0.6rem; }
.ring { flex: none; width: 128px; height: 128px; border-radius: 50%; display: grid; place-items: center; }
.ring > div { width: 98px; height: 98px; border-radius: 50%; background: #fff; display: flex;
  flex-direction: column; align-items: center; justify-content: center; }
.ring b { font-size: 2.3rem; font-weight: 800; line-height: 1; }
.ring small { color: $MUTED; font-size: 0.78rem; }
.verdict .lv { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.05; }
.verdict .txt { color: $INK; font-size: 1rem; margin-top: 0.45rem; max-width: 50ch; line-height: 1.55; }
.cards { display: grid; gap: 0.9rem; grid-template-columns: repeat(2, 1fr); }
.card { background: #fff; border-radius: 14px; padding: 0.95rem 1.1rem; border-left: 5px solid var(--c); }
.card .h { font-weight: 800; color: $INK; }
.card .d { color: $MUTED; font-size: 0.93rem; margin-top: 0.25rem; line-height: 1.5; }
.pill { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-weight: 700;
  font-size: 0.85rem; }
.note { font-size: 0.85rem; color: $MUTED; }
.foot { margin-top: 2rem; font-size: 0.82rem; color: $MUTED; }
[data-testid="stMetricValue"] { font-weight: 800; color: $INK; }
@media (max-width: 640px) {
  .steps, .cards { grid-template-columns: 1fr; }
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


def score_scale(thr):
    """Green below the borderline band, amber in it, red at or above the threshold."""
    t = thr * 100
    return alt.Scale(domain=[0, max(t - 10, 0.1), t, 100],
                     range=[C["LOW"], C["LOW"], C["MID"], C["HIGH"]])


SENSOR = {"acc": "Acceleration", "gyro": "Rotation"}
STAT = {"mean": "average", "std": "variability", "rms": "intensity", "range": "range",
        "skew": "skew", "kurt": "peakiness"}
SPECIAL = {"jerk_mean": "Jerk, average", "jerk_std": "Jerk, variability",
           "jerk_rms": "Jerk, intensity", "sma": "Overall movement (SMA)",
           "mean_hr": "Heart rate, average", "mean_ibi": "Beat interval, average"}
SPECTRAL = {"dom_freq": "dominant rhythm", "spectral_entropy": "irregularity",
            "low_freq_power": "slow-movement power", "high_freq_power": "fast-movement power",
            "spectral_centroid": "frequency centre"}


def pretty(f):
    """Readable name for a feature column."""
    if f in SPECIAL:
        return SPECIAL[f]
    for k, v in SPECTRAL.items():
        if f.startswith(k + "_"):
            return f"{SENSOR.get(f.rsplit('_', 1)[1], '')} {v}".strip().capitalize()
    p = f.split("_")
    if len(p) == 3 and p[1] == "corr":
        return f"{SENSOR.get(p[0], p[0])} {p[2][0]}–{p[2][1]} coupling"
    if len(p) == 3 and p[0] in SENSOR:
        axis = "magnitude" if p[1] == "mag" else f"{p[1]}-axis"
        return f"{SENSOR[p[0]]} {axis}, {STAT.get(p[2], p[2])}"
    return f.replace("_", " ")


def footer_and_settings():
    with st.expander("Model settings"):
        st.text_input("Model folder", key="model_dir")
    st.markdown("<div class='foot'>Research prototype, trained on a small study of 14 people. "
                "Not a medical device and not a diagnosis.</div>", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────
st.markdown("""<div class='hero'>
<h1>See when fatigue set in during desk work</h1>
<p>Upload an Apple Watch session, then explore the fatigue score over time: zoom into any
stretch, inspect single moments, and compare parts of the session.</p>
</div>""", unsafe_allow_html=True)

try:
    models = get_models(st.session_state.model_dir)
except Exception as err:
    st.error(f"Couldn't load the models. {err}")
    st.text_input("Model folder", key="model_dir")
    st.stop()
model_thr = float(models["threshold"])

uploads = st.file_uploader("Upload a session", type=["zip", "csv"],
                           accept_multiple_files=True, label_visibility="collapsed")
if uploads:
    st.session_state.demo = False

if not uploads and not st.session_state.get("demo"):
    st.markdown(Template("""<div class='steps'>
<div class='step' style='--c:$INDIGO'><span class='n'>1</span><div class='t'>Record</div>
<div class='d'>Wear the watch and run the EduFatigue app during desk work.</div></div>
<div class='step' style='--c:$LOW'><span class='n'>2</span><div class='t'>Upload</div>
<div class='d'>Drop the zipped session folder, or its CSV files, above.</div></div>
<div class='step' style='--c:$MID'><span class='n'>3</span><div class='t'>Explore</div>
<div class='d'>Zoom, hover and pick moments to see how fatigue changed.</div></div>
</div>""").substitute(C), unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2], vertical_alignment="center")
    if c1.button("Try a demo session", type="primary", width="stretch"):
        st.session_state.demo = True
        st.rerun()
    c2.caption("**Accelerometer.csv** is required. **Gyroscope.csv** and "
               "**HeartRate_IBI.csv** make the estimate better.")
    footer_and_settings()
    st.stop()

if st.session_state.get("demo") and not uploads:
    acc, gyr, hr = make_demo_session()
    top = st.columns([4, 1], vertical_alignment="center")
    top[0].info("Demo: a synthetic session. Its result says nothing about a real person.")
    if top[1].button("Exit demo", width="stretch"):
        st.session_state.demo = False
        st.rerun()
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
proba = np.asarray(res["proba"])
dur = float(starts[-1] + WIN_MIN)

has_hr = hr is not None and {"timestamp_unix_ms", "bpm"}.issubset(hr.columns) \
    and hr["bpm"].notna().sum() > 1
if has_hr:
    hr_df = hr.dropna(subset=["bpm"]).copy()
    hr_df["Minute"] = (hr_df["timestamp_unix_ms"] - hr_df["timestamp_unix_ms"].iloc[0]) / 60000
    hr_df = hr_df[["Minute", "bpm"]]

# ── Controls ─────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("**Explore this session**")
    k1, k2 = st.columns(2)
    lo, hi = k1.slider("Minutes to analyse", 0.0, round(dur, 1), (0.0, round(dur, 1)),
                       step=0.5, format="%.1f min",
                       help="Everything below is recalculated for this stretch only.")
    thr = k2.slider("Fatigue threshold", 5, 95, int(round(model_thr * 100)), step=1,
                    help=f"The model's own threshold is {model_thr * 100:.0f}. Moving this "
                         "is a what-if: it changes which windows count as fatigued, "
                         "not the model.") / 100
    k3, k4 = st.columns(2)
    smooth_opt = k3.segmented_control("Smoothing", ["None", "1 min", "3 min"],
                                      default="1 min", key="smooth")
    split_opt = k4.segmented_control("Compare by", ["Thirds", "5-min blocks", "10-min blocks"],
                                     default="Thirds", key="split")
    if thr != round(model_thr, 2):
        st.markdown(f"<span class='note'>What-if threshold {thr * 100:.0f} "
                    f"(model default {model_thr * 100:.0f}).</span>", unsafe_allow_html=True)

win = {"None": 1, "1 min": 6, "3 min": 18}[smooth_opt or "1 min"]
smooth = pd.Series(proba).rolling(win, min_periods=1, center=True).mean().values

mask = (starts >= lo) & (starts + WIN_MIN <= hi + 1e-9)
if mask.sum() < 3:
    st.warning("Pick a stretch of at least one minute.")
    st.stop()
idx = np.flatnonzero(mask)
p_sel, s_sel, t_sel = proba[idx], smooth[idx], starts[idx]

# ── Verdict ──────────────────────────────────────────────────────────
score = float(p_sel.mean())
level, color, tint = level_of(score, thr)
above = p_sel >= thr
onset, run = None, 0
for i, a in enumerate(above):
    run = run + 1 if a else 0
    if run >= 3:
        onset = t_sel[i - 2]
        break
share = above.mean() * 100
scope = "this session" if (lo == 0 and hi >= round(dur, 1)) else f"minutes {lo:.0f}–{hi:.0f}"
when = (f"sustained from minute {onset:.0f}" if onset is not None
        else "never for a sustained stretch")
st.markdown(f"""<div class='verdict' style='background:{tint}'>
<div class='ring' style='background:conic-gradient({color} {score * 100:.0f}%, #ffffff 0)'>
<div><b style='color:{color}'>{score * 100:.0f}</b><small>of 100</small></div></div>
<div><div class='lv' style='color:{color}'>{level}</div>
<div class='txt'>Across {scope}, {share:.0f}% of the 20-second windows were at or above the
threshold, {when}.</div></div></div>""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Length analysed", f"{hi - lo:.0f} min")
m2.metric("Windows", f"{len(idx)}")
m3.metric("Peak score", f"{s_sel.max() * 100:.0f}",
          help="Highest smoothed score in the stretch.")
m4.metric("Peak at", f"min {t_sel[int(np.argmax(s_sel))]:.0f}")

# ── Interactive timeline ─────────────────────────────────────────────
st.markdown("### Timeline")
st.caption("Drag across the colour strip at the bottom to zoom all charts; double-click it "
           "to reset. Hover any chart for exact values.")

df = pd.DataFrame({"Minute": t_sel, "Score": p_sel * 100, "Smoothed": s_sel * 100})
df["Level"] = [level_of(p, thr)[0] for p in p_sel]
has_move = "acc_mag_std" in feats.columns
if has_move:
    df["Movement"] = feats["acc_mag_std"].values[idx]

x_dom = [float(t_sel[0]), float(t_sel[-1] + WIN_MIN)]
brush = alt.selection_interval(encodings=["x"], name="zoom")
x_zoom = alt.X("Minute:Q", title=None, scale=alt.Scale(domain=brush),
               axis=alt.Axis(labels=False, ticks=False))
hover = alt.selection_point(fields=["Minute"], nearest=True, on="pointerover",
                            empty=False, clear="pointerout", name="hover")

base = alt.Chart(df)
raw = base.mark_line(color=C["LINE"], strokeWidth=1, clip=True).encode(
    x=x_zoom, y=alt.Y("Score:Q", title="Fatigue score", scale=alt.Scale(domain=[0, 100])))
line = base.mark_line(color=C["INDIGO"], strokeWidth=3, clip=True).encode(x=x_zoom, y="Smoothed:Q")
band = alt.Chart(pd.DataFrame({"y0": [max(thr * 100 - 10, 0)], "y1": [thr * 100]})).mark_rect(
    color=C["MID"], opacity=0.12).encode(y="y0:Q", y2="y1:Q")
rule = alt.Chart(pd.DataFrame({"t": [thr * 100]})).mark_rule(
    color=C["HIGH"], strokeDash=[5, 4]).encode(y="t:Q")
dots = base.mark_circle(size=90, clip=True).encode(
    x=x_zoom, y="Smoothed:Q",
    color=alt.Color("Smoothed:Q", scale=score_scale(thr), legend=None),
    opacity=alt.condition(hover, alt.value(1), alt.value(0)),
    tooltip=[alt.Tooltip("Minute:Q", format=".1f"),
             alt.Tooltip("Smoothed:Q", title="Score (smoothed)", format=".0f"),
             alt.Tooltip("Score:Q", title="Score (this window)", format=".0f"),
             "Level:N"]).add_params(hover)
vline = base.mark_rule(color=C["MUTED"], strokeDash=[2, 2]).encode(x=x_zoom).transform_filter(hover)
score_chart = (band + raw + line + rule + vline + dots).properties(height=230)

rows = [score_chart]
if has_move:
    rows.append(base.mark_line(color=C["VIOLET"], strokeWidth=2, clip=True).encode(
        x=x_zoom, y=alt.Y("Movement:Q", title="Wrist motion", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("Minute:Q", format=".1f"),
                 alt.Tooltip("Movement:Q", title="Motion variability", format=".3f")]
    ).properties(height=90))
if has_hr:
    h_sel = hr_df[(hr_df.Minute >= lo) & (hr_df.Minute <= hi)]
    rows.append(alt.Chart(h_sel).mark_line(color=C["ROSE"], strokeWidth=1.5, clip=True).encode(
        x=x_zoom, y=alt.Y("bpm:Q", title="Heart rate", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("Minute:Q", format=".1f"), alt.Tooltip("bpm:Q", format=".0f")]
    ).properties(height=90))

strip = alt.Chart(df.assign(End=df.Minute + WIN_MIN)).mark_rect().encode(
    x=alt.X("Minute:Q", title="Minute of session", scale=alt.Scale(domain=x_dom, nice=False)),
    x2="End:Q",
    color=alt.Color("Score:Q", scale=score_scale(thr), legend=None),
    tooltip=[alt.Tooltip("Minute:Q", format=".1f"), alt.Tooltip("Score:Q", format=".0f")]
).add_params(brush).properties(height=40)
rows.append(strip)

timeline = alt.vconcat(*rows, spacing=6).configure_view(stroke=None).configure_axis(
    labelColor=C["MUTED"], titleColor=C["MUTED"], gridColor="#EEF1F7")
st.altair_chart(timeline, width="stretch")

# ── Compare parts ────────────────────────────────────────────────────
st.markdown("### How it changed")
split = split_opt or "Thirds"
if split == "Thirds":
    groups = np.array_split(np.arange(len(idx)), 3)
    labels = ["Start", "Middle", "End"]
else:
    size = 5 if split.startswith("5") else 10
    b = ((t_sel - lo) // size).astype(int)
    groups = [np.flatnonzero(b == k) for k in np.unique(b)]
    labels = [f"{lo + k * size:.0f}–{min(lo + (k + 1) * size, hi):.0f}" for k in np.unique(b)]
blocks = pd.DataFrame({
    "Part": labels,
    "Score": [p_sel[g].mean() * 100 for g in groups],
    "From": [t_sel[g[0]] for g in groups],
    "To": [t_sel[g[-1]] + WIN_MIN for g in groups],
    "Fatigued windows": [f"{(p_sel[g] >= thr).mean() * 100:.0f}%" for g in groups]})
blocks["Level"] = [level_of(s / 100, thr)[0] for s in blocks.Score]
bars = alt.Chart(blocks).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
    x=alt.X("Part:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
    y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 100]), title="Mean score"),
    color=alt.Color("Score:Q", scale=score_scale(thr), legend=None),
    tooltip=["Part", alt.Tooltip("From:Q", format=".1f", title="From minute"),
             alt.Tooltip("To:Q", format=".1f", title="To minute"),
             alt.Tooltip("Score:Q", format=".0f"), "Level", "Fatigued windows"])
labels_ = bars.mark_text(dy=-8, fontWeight="bold", color=C["INK"]).encode(
    text=alt.Text("Score:Q", format=".0f"))
brule = alt.Chart(pd.DataFrame({"t": [thr * 100]})).mark_rule(
    color=C["HIGH"], strokeDash=[5, 4]).encode(y="t:Q")
st.altair_chart((bars + labels_ + brule).properties(height=220).configure_view(stroke=None),
                width="stretch")

# ── Moment inspector ─────────────────────────────────────────────────
st.markdown("### Inspect a moment")
opts = [f"{m:.1f}" for m in t_sel]
peak_i = int(np.argmax(s_sel))
pick = st.select_slider("Minute", options=opts, value=opts[peak_i],
                        help="Starts at the peak. Slide to any 20-second window.")
j = opts.index(pick)
gi = idx[j]
lv, lc, lt = level_of(proba[gi], thr)
i1, i2 = st.columns([1, 2], gap="large")
with i1:
    st.markdown(f"<div class='note'>Window starting at minute {t_sel[j]:.1f}</div>"
                f"<div style='font-size:2.6rem;font-weight:800;color:{lc};line-height:1.1'>"
                f"{proba[gi] * 100:.0f}</div>"
                f"<span class='pill' style='background:{lt};color:{lc}'>{lv}</span>",
                unsafe_allow_html=True)
    pm = pd.DataFrame({"Model": [NAMES[k] for k in WEIGHTS],
                       "Score": [float(res["per_model"][k][gi]) * 100 for k in WEIGHTS]})
    st.altair_chart(alt.Chart(pm).mark_bar(height=14, cornerRadiusEnd=4).encode(
        x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 100]), title="Each model's score"),
        y=alt.Y("Model:N", title=None, sort=None),
        color=alt.Color("Score:Q", scale=score_scale(thr), legend=None),
        tooltip=["Model", alt.Tooltip("Score:Q", format=".0f")]
    ).properties(height=110).configure_view(stroke=None), width="stretch")
with i2:
    used = [f for f in models["features"] if f in feats.columns]
    F = feats[used].iloc[idx]
    sd = F.std().replace(0, np.nan)
    z = ((F.iloc[j] - F.mean()) / sd).dropna()
    top = z.reindex(z.abs().sort_values(ascending=False).index)[:6]
    if len(top):
        zdf = pd.DataFrame({"Feature": [pretty(f) for f in top.index], "z": top.values})
        zdf["Direction"] = np.where(zdf.z > 0, "Higher than usual", "Lower than usual")
        st.altair_chart(alt.Chart(zdf).mark_bar(height=16, cornerRadius=3).encode(
            x=alt.X("z:Q", title="Difference from this session's average (SD)"),
            y=alt.Y("Feature:N", sort=None, title=None),
            color=alt.Color("Direction:N", scale=alt.Scale(
                domain=["Higher than usual", "Lower than usual"],
                range=[C["VIOLET"], C["LOW"]]), legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Feature", alt.Tooltip("z:Q", format="+.1f", title="SD from average")]
        ).properties(height=210).configure_view(stroke=None), width="stretch")
        st.markdown("<p class='note'>What was unusual in this window compared with the rest "
                    "of the stretch. It describes the signal, not the model's reasoning.</p>",
                    unsafe_allow_html=True)

# ── What stood out ───────────────────────────────────────────────────
thirds = np.array_split(np.arange(len(idx)), 3)
cards = [(C["MID"], "⏱️ Fatigue set in",
          f"Around minute {onset:.0f}; three windows in a row crossed the threshold."
          if onset is not None else "Not at any sustained point in this stretch."),
         (C["HIGH"], "📈 Peak",
          f"Highest around minute {t_sel[peak_i]:.0f} (score {s_sel[peak_i] * 100:.0f}).")]
if has_move and len(idx) >= 6:
    mv = df["Movement"].values
    a0, a1 = mv[thirds[0]].mean(), mv[thirds[2]].mean()
    if a0 > 0:
        r = (a1 - a0) / a0 * 100
        cards.append((C["VIOLET"], "🖐️ Movement",
                      f"Wrist movement was {abs(r):.0f}% {'more' if r > 0 else 'less'} "
                      "variable in the last third than in the first."))
if has_hr:
    h = hr_df[(hr_df.Minute >= lo) & (hr_df.Minute <= hi)]["bpm"]
    if len(h) > 6:
        k = max(len(h) // 3, 1)
        d = h.iloc[-k:].mean() - h.iloc[:k].mean()
        cards.append((C["ROSE"], "❤️ Heart rate",
                      f"{abs(d):.0f} bpm {'higher' if d > 0 else 'lower'} in the last third "
                      f"than in the first (range {h.min():.0f}–{h.max():.0f})."))
st.markdown("### What stood out")
st.markdown("<div class='cards'>" + "".join(
    f"<div class='card' style='--c:{c}'><div class='h'>{h_}</div><div class='d'>{d_}</div></div>"
    for c, h_, d_ in cards) + "</div>", unsafe_allow_html=True)

# ── Details ──────────────────────────────────────────────────────────
st.markdown("### Details")
t1, t2, t3 = st.tabs(["Model agreement", "Data quality", "Footprint"])
with t1:
    long = pd.DataFrame({"Minute": np.tile(t_sel, len(WEIGHTS)),
                         "Model": np.repeat([NAMES[k] for k in WEIGHTS], len(idx)),
                         "Score": np.concatenate([pd.Series(res["per_model"][k][idx]).rolling(
                             win, min_periods=1, center=True).mean() * 100 for k in WEIGHTS])})
    pick_m = alt.selection_point(fields=["Model"], bind="legend")
    st.altair_chart(alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("Minute:Q", title="Minute of session"),
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 100]), title="Score"),
        color=alt.Color("Model:N", scale=alt.Scale(range=[C["INDIGO"], C["VIOLET"], C["LOW"]]),
                        legend=alt.Legend(orient="bottom", title=None)),
        opacity=alt.condition(pick_m, alt.value(1), alt.value(0.12)),
        tooltip=["Model", alt.Tooltip("Minute:Q", format=".1f"),
                 alt.Tooltip("Score:Q", format=".0f")]
    ).add_params(pick_m).properties(height=240).configure_view(stroke=None), width="stretch")
    wtxt = ", ".join(f"{NAMES[k]} {v:.2f}" for k, v in WEIGHTS.items())
    st.markdown(f"<p class='note'>Click a model in the legend to highlight it. The final score "
                f"is a weighted average: {wtxt}.</p>", unsafe_allow_html=True)
with t2:
    q1, q2, q3 = st.columns(3)
    q1.metric("Recording", f"{info['duration_min']:.1f} min")
    q2.metric("Gyroscope", "Used" if info["gyro_used"] else "Missing")
    q3.metric("Heart-rate samples", f"{len(hr_df) if has_hr else 0}")
    if not info["gyro_used"]:
        st.warning("Gyroscope data was missing or too short, so its features were set to zero.")
    if not info["hrv"]:
        st.warning("Too few heart-rate samples, so heart-rate features were set to zero.")
    st.markdown("<p class='note'>The watch reports heart rate about every 5 seconds, so "
                "beat-to-beat variability (HRV) cannot be measured from this data.</p>",
                unsafe_allow_html=True)
with t3:
    a, b, c = st.columns(3)
    a.metric("Model size", f"{models['size_mb']:.1f} MB")
    b.metric("Features per window", f"{info['feature_ms_per_window']:.1f} ms")
    c.metric("Inference per window", f"{res['infer_ms_per_window']:.1f} ms")
    st.markdown("<p class='note'>Measured on the machine running this app, not on the watch.</p>",
                unsafe_allow_html=True)

table = pd.DataFrame({
    "start_min": np.round(t_sel, 2), "fatigue_score": np.round(p_sel * 100, 1),
    "smoothed": np.round(s_sel * 100, 1),
    **{NAMES[k].lower().replace(" ", "_"): np.round(res["per_model"][k][idx] * 100, 1)
       for k in WEIGHTS}})
st.download_button(f"Download scores for minutes {lo:.0f}–{hi:.0f} (CSV)",
                   table.to_csv(index=False), file_name="edgefatigue_scores.csv",
                   mime="text/csv")
footer_and_settings()
