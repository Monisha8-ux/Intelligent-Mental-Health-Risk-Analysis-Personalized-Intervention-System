# ── MENTAL HEALTH RISK ANALYSIS — Streamlit App ────────────────────────────

import streamlit as st
import pandas   as pd
import numpy    as np
import joblib, os
from encoders import OrdinalMapEncoder

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Mental Health Risk Analyzer",
    page_icon  = "🧠",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: linear-gradient(160deg, #0d0d1a 0%, #121228 50%, #0d1a24 100%);
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: #0f111e !important;
    border-right: 1px solid rgba(99,102,241,0.25);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h2 { color: #a5b4fc !important; font-size: 1.1rem !important; }
[data-testid="stSidebar"] strong { color: #e2e8f0 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(99,102,241,0.3) !important; }

.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99,102,241,0.20);
    border-radius: 14px;
    padding: 1.5rem 1.6rem 1.2rem;
    margin-bottom: 0;
}
.card-title {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(99,102,241,0.20);
}

.hero {
    background: linear-gradient(135deg,
        rgba(99,102,241,0.18) 0%,
        rgba(139,92,246,0.12) 50%,
        rgba(14,165,233,0.10) 100%);
    border: 1px solid rgba(99,102,241,0.30);
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    text-align: center;
    margin-bottom: 1.8rem;
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 0.5rem;
    background: linear-gradient(90deg, #818cf8, #c084fc, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p { color: #94a3b8; font-size: 1rem; margin: 0; }

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 1.5rem !important;
    font-size: 1.02rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px !important;
    transition: all 0.25s !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.55) !important;
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
}

label, .stSelectbox label, .stRadio label, .stSlider label {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
.stRadio > div { gap: 0.5rem; }
.stRadio > div > label { color: #cbd5e1 !important; }

.badge {
    display: inline-block;
    padding: 0.45rem 1.4rem;
    border-radius: 50px;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    margin-top: 0.6rem;
}
.b0 { background: linear-gradient(135deg,#991b1b,#7f1d1d); color:#fff; }
.b1 { background: linear-gradient(135deg,#c2410c,#7c2d12); color:#fff; }
.b2 { background: linear-gradient(135deg,#92400e,#78350f); color:#fff; }
.b3 { background: linear-gradient(135deg,#166534,#14532d); color:#fff; }
.b4 { background: linear-gradient(135deg,#1e40af,#1e3a8a); color:#fff; }

.prob-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.55rem;
}
.prob-label  { width: 90px; font-size: 0.82rem; color: #94a3b8; }
.prob-track  { flex: 1; background: #1e293b; border-radius: 999px; height: 12px; overflow:hidden; }
.prob-fill   { height: 100%; border-radius: 999px;
               background: linear-gradient(90deg,#6366f1,#a78bfa); }
.prob-fill-dim{ height:100%; border-radius:999px; background:#334155; }
.prob-pct    { width: 40px; text-align: right; font-size: 0.80rem; color: #64748b; }

.rec {
    background: rgba(99,102,241,0.07);
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 0.65rem 0.9rem;
    margin: 0.45rem 0;
    font-size: 0.92rem;
    line-height: 1.55;
    color: #cbd5e1;
}

.risk-alert {
    background: rgba(239,68,68,0.10);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-top: 0.8rem;
    font-size: 0.88rem;
    color: #fca5a5;
}

.sum-row { display:flex; justify-content:space-between; padding:0.4rem 0;
           border-bottom:1px solid rgba(255,255,255,0.05); }
.sum-key { color:#64748b; font-size:0.83rem; }
.sum-val { color:#e2e8f0; font-size:0.83rem; font-weight:500; max-width:55%; text-align:right; }

.divider { border: none; border-top: 1px solid rgba(99,102,241,0.18); margin: 1.4rem 0; }
</style>
""", unsafe_allow_html=True)


# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = "mental_health_model.pkl"
    if not os.path.exists(path):
        st.error("Model not found. Run `mental_health_model.py` first.")
        st.stop()
    return joblib.load(path)

artefacts      = load_model()
model          = artefacts["model"]
label_encoders = artefacts["label_encoders"]
features       = artefacts["features"]
class_names    = ["Very Bad", "Bad", "Normal", "Good", "Very Healthy"]


# ── ENCODING HELPER ───────────────────────────────────────────────────────────
def encode_input(user_data: dict) -> np.ndarray:
    base_features = features[:12]
    row = {}
    for feat in base_features:
        enc   = label_encoders[feat]
        value = str(user_data[feat])
        if hasattr(enc, "_map"):
            row[feat] = enc.transform([value])[0]
        elif value in enc.classes_:
            row[feat] = enc.transform([value])[0]
        else:
            norm    = value.encode("ascii", "ignore").decode()
            matched = False
            for cls in enc.classes_:
                if str(cls).encode("ascii", "ignore").decode() == norm:
                    row[feat] = enc.transform([cls])[0]
                    matched   = True
                    break
            if not matched:
                row[feat] = 0

    stress_enc = row["stress_experience"]
    mood_enc   = row["avg_mood"]

    sleep_str = str(user_data["sleep_hours"])
    if   "Less" in sleep_str:    sleep_ord = 0
    elif "5" in sleep_str[:3]:   sleep_ord = 1
    elif "7" in sleep_str[:3]:   sleep_ord = 2
    else:                        sleep_ord = 3

    row["stress_x_mood"]  = stress_enc * mood_enc
    row["sleep_x_stress"] = sleep_ord  * (3 - stress_enc)

    return np.array([[row[f] for f in features]])


# ── RULE-BASED SAFETY NET ─────────────────────────────────────────────────────
def apply_safety_rules(pred_cls: int, user_data: dict) -> tuple[int, list[str]]:
    warnings = []
    sm     = str(user_data.get("daily_sm_hours", ""))
    sleep  = str(user_data.get("sleep_hours", ""))
    stress = str(user_data.get("stress_experience", ""))
    mood   = str(user_data.get("avg_mood", ""))

    bad_sleep = "Less" in sleep
    high_sm   = "5+" in sm
    mid_sm    = "3" in sm[:2]

    if bad_sleep and high_sm and pred_cls >= 2:
        pred_cls = min(pred_cls, 1)
        warnings.append("⚠️ Less than 5 hrs sleep combined with 5+ hrs social media is a serious risk pattern.")

    if bad_sleep and (high_sm or mid_sm) and stress == "Always" and pred_cls >= 1:
        pred_cls = 0
        warnings.append("⚠️ Chronic stress, high social media usage, and poor sleep strongly indicate very bad mental health.")

    if stress == "Always" and mood == "Sad" and pred_cls >= 2:
        pred_cls = 1
        warnings.append("⚠️ Constant stress and persistently sad mood are strong indicators of bad mental health.")

    if bad_sleep and pred_cls >= 3:
        pred_cls = 2
        warnings.append("⚠️ Sustained sleep deprivation (< 5 hrs) limits overall well-being to Normal at best.")

    if high_sm and pred_cls == 4:
        pred_cls = 3
        warnings.append("ℹ️ Heavy social media use (5+ hrs/day) is associated with reduced overall well-being.")

    return pred_cls, warnings


# ── RECOMMENDATIONS ───────────────────────────────────────────────────────────
RECOMMENDATIONS = {
    0: {
        "emoji": "🚨",
        "label": "Very Bad Mental Health",
        "badge": "b0",
        "summary": "Your responses indicate **critically poor mental health**. Immediate professional support is strongly advised.",
        "recs": [
            "📞 Contact a mental health professional or counsellor as soon as possible — this is urgent.",
            "🛑 Limit daily social media to under 30 minutes or delete apps temporarily to break the cycle.",
            "😴 Start a fixed sleep schedule tonight: aim for 7–9 hours and stop screen use 1 hour before bed.",
            "🧘 Practice daily grounding techniques — box breathing, body-scan meditation, or cold water splash.",
            "🚶 Walk outside for at least 15 minutes every day — even this measurably lowers cortisol.",
            "📔 Keep a daily mood journal to identify your emotional triggers and track progress.",
            "👥 Open up to at least one trusted person about how you are genuinely feeling.",
            "🔕 Disable all social media notifications and unfollow accounts that trigger comparison.",
        ],
    },
    1: {
        "emoji": "⚠️",
        "label": "Bad Mental Health",
        "badge": "b1",
        "summary": "Your well-being is noticeably low. Consistent, targeted actions can make a real difference quickly.",
        "recs": [
            "⏰ Set a strict 1–2 hour daily cap on social media — use your phone's built-in screen-time tools.",
            "🌙 Stop all social media 45 minutes before sleep and replace it with reading or light stretching.",
            "🏋️ Begin a simple 3-day-per-week exercise habit — physical movement reduces stress hormones directly.",
            "🧠 Use a guided meditation app (Headspace, Calm) for just 10 minutes each morning.",
            "😤 Write down your top 3 stress triggers and one concrete coping strategy for each.",
            "🥗 Cut caffeine after 2 PM and drink more water — both significantly stabilise mood.",
            "📵 Make your bedroom a phone-free zone — charge it outside the room overnight.",
            "🗣️ Consider booking even one session with a counsellor to get started.",
        ],
    },
    2: {
        "emoji": "🔶",
        "label": "Normal Mental Health",
        "badge": "b2",
        "summary": "You're getting by, but real balance is within reach. Small consistent changes will lift you noticeably.",
        "recs": [
            "📊 Do a weekly social media check-in — identify which apps drain your energy and reduce them first.",
            "🏃 Target 150 minutes of moderate exercise per week (e.g., 30 min brisk walk × 5 days).",
            "😴 Lock in a consistent sleep and wake time — irregular sleep quietly wrecks mood stability.",
            "📖 Replace 30 minutes of daily scrolling with one intentional activity you enjoy or want to learn.",
            "🎯 Set one small, meaningful personal goal each week to feel a sense of progress.",
            "🧘 Try the 5-4-3-2-1 grounding technique whenever you feel stressed or scattered.",
            "🤝 Plan regular face-to-face time with friends or family — quality beats quantity.",
            "📵 Clean up your social feeds: unfollow anything that makes you feel worse after viewing.",
        ],
    },
    3: {
        "emoji": "✅",
        "label": "Good Mental Health",
        "badge": "b3",
        "summary": "You have a solid mental health foundation. Stay consistent and keep building on these habits.",
        "recs": [
            "🌿 Keep your current routines — the compounding power of consistent good habits is enormous.",
            "📚 Invest in a challenging new hobby that exercises your mind and sparks curiosity.",
            "🤲 Consider volunteering or mentoring someone — giving back is one of the strongest well-being boosters.",
            "📝 Review monthly what habits are working and actively protect them during busy or stressful periods.",
            "🏕️ Spend time in nature regularly — even 20 minutes in a green space measurably reduces cortisol.",
            "🔋 Rest before you are exhausted — proactive rest is a performance strategy, not laziness.",
            "🎵 Use social media intentionally and creatively rather than through passive habitual scrolling.",
            "💬 Keep conversations about emotional health open with the people close to you.",
        ],
    },
    4: {
        "emoji": "🌟",
        "label": "Very Healthy Mental Health",
        "badge": "b4",
        "summary": "Your mental health is excellent! Focus on sustainability and inspiring the people around you.",
        "recs": [
            "🏆 Document your positive routines so you can find your way back to them during hard periods.",
            "🧩 Take on a genuinely stretching challenge — a new language, skill, or creative project.",
            "🤲 Share your experience openly — coaching or mentoring others strongly reinforces your own well-being.",
            "⚖️ Watch for gradual burnout proactively; high-functioning people are most vulnerable to it creeping in.",
            "🌍 Widen your social circle — encountering diverse people and ideas builds long-term resilience.",
            "📓 Keep a gratitude or reflection journal to maintain and deepen your self-awareness.",
            "🔄 Audit your social media diet every few months — intentional usage beats habitual usage every time.",
            "🌱 Set ambitious long-term life goals and turn them into a series of small, trackable milestones.",
        ],
    },
}


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 About This App")
    st.markdown("""
Analyses your **social media habits & lifestyle** to estimate mental health
risk using a trained **XGBoost classifier** combined with clinical logic rules.

---

**How it works**
1. Fill in the form
2. Click **Analyse My Mental Health**
3. Receive a risk rating + personalised recommendations

---

**5 Risk Levels**

| Level | Meaning |
|---|---|
| 🚨 Very Bad | Immediate action needed |
| ⚠️ Bad | Below average |
| 🔶 Normal | Managing, room to grow |
| ✅ Good | Solid foundation |
| 🌟 Very Healthy | Thriving |

---

**Model info**
- Algorithm: XGBoost (5-class)
- Training data: 1,361 survey responses
- Features: 14 (incl. interaction)
- Logic safety rules: enabled

---

> **Disclaimer** — Educational ML tool, not a medical diagnosis.
> If in distress, please contact a qualified mental health professional.
""")


# ── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧠 Mental Health Risk Analyzer</h1>
    <p>AI-powered analysis of your social media habits &amp; lifestyle indicators</p>
</div>
""", unsafe_allow_html=True)


# ── INPUT FORM ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown('<div class="card"><div class="card-title">👤 Demographics</div>', unsafe_allow_html=True)
    age = st.selectbox("Age Group", ["Under 18", "18–22", "23–30", "30+"], index=1, key="age")
    gender = st.selectbox("Gender", ["Female", "Male", "Prefer not to say"], index=0, key="gender")
    occupation = st.selectbox(
        "Occupation",
        ["Student", "Working (Full-time/Part-time)", "Other (Unemployed, Retired, Homemaker, etc.)"],
        index=0, key="occupation"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><div class="card-title">📱 Social Media Habits</div>', unsafe_allow_html=True)
    daily_sm_hours = st.selectbox(
        "Daily Social Media Usage",
        ["0–1 hrs", "1–3 hrs", "3–5 hrs", "5+ hrs"],
        index=1, key="daily_sm_hours"
    )
    peak_usage_time = st.selectbox(
        "Peak Usage Time",
        [
            "Morning (Waking up until 12 PM)",
            "Afternoon (12 PM – 5 PM)",
            "Night (5 PM – 10 PM)",
            "Late Night (10 PM onwards)",
        ],
        index=3, key="peak_usage_time"
    )
    before_sleep_usage = st.radio(
        "Social media before sleep?",
        ["Yes", "No"],
        horizontal=True, key="before_sleep_usage"
    )
    purpose = st.selectbox(
        "Main Purpose",
        [
            "Entertainment (e.g., watching videos, memes)",
            "Timepass/Habitual Scrolling",
            "Communication (e.g., direct messaging, group chats)",
            "Learning/Information (e.g., news, tutorials)",
            "Professional/Work Networking",
        ],
        index=0, key="purpose"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card"><div class="card-title">🌡️ Lifestyle & Mental Indicators</div>', unsafe_allow_html=True)
    sleep_hours = st.selectbox(
        "Sleep Duration per Night",
        ["Less than 5 hrs", "5–7 hrs", "7–9 hrs", "9+ hrs"],
        index=2, key="sleep_hours"
    )
    stress_experience = st.selectbox(
        "Stress / Anxiety Frequency",
        ["Never", "Sometimes", "Often", "Always"],
        index=1, key="stress_experience"
    )
    avg_mood = st.selectbox(
        "General Mood on Average Day",
        ["Happy", "Neutral", "Sad"],
        index=1, key="avg_mood"
    )
    productivity_impact = st.selectbox(
        "Social Media & Productivity",
        [
            "No, it has no major impact",
            "Maybe, sometimes it helps, sometimes it hinders",
            "Yes, significantly reduces it",
        ],
        index=1, key="productivity_impact"
    )
    comparison_frequency = st.selectbox(
        "Compare Yourself with Others on Social Media?",
        ["Never", "Sometimes", "Often", "Always"],
        index=1, key="comparison_frequency"
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ── PREDICT BUTTON ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1.5, 1, 1.5])
with btn_col:
    analyse = st.button("🔍  Analyse My Mental Health", use_container_width=True)


# ── DISPLAY MAPPER ────────────────────────────────────────────────────────────
def resolve(display_val: str, feat: str) -> str:
    enc = label_encoders[feat]
    if hasattr(enc, "_map"):
        return display_val
    if display_val in enc.classes_:
        return display_val
    norm = display_val.encode("ascii", "ignore").decode()
    for cls in enc.classes_:
        if str(cls).encode("ascii", "ignore").decode() == norm:
            return cls
    return str(enc.classes_[0])


# ── PREDICTION & RESULTS ──────────────────────────────────────────────────────
if analyse:
    user = {
        "age":                  resolve(age,                   "age"),
        "occupation":           resolve(occupation,            "occupation"),
        "gender":               resolve(gender,                "gender"),
        "daily_sm_hours":       resolve(daily_sm_hours,        "daily_sm_hours"),
        "peak_usage_time":      resolve(peak_usage_time,       "peak_usage_time"),
        "before_sleep_usage":   resolve(before_sleep_usage,    "before_sleep_usage"),
        "purpose":              resolve(purpose,               "purpose"),
        "sleep_hours":          resolve(sleep_hours,           "sleep_hours"),
        "stress_experience":    resolve(stress_experience,     "stress_experience"),
        "avg_mood":             resolve(avg_mood,              "avg_mood"),
        "productivity_impact":  resolve(productivity_impact,   "productivity_impact"),
        "comparison_frequency": resolve(comparison_frequency,  "comparison_frequency"),
    }

    X        = encode_input(user)
    raw_cls  = int(model.predict(X)[0])
    proba    = model.predict_proba(X)[0]

    final_cls, rule_warnings = apply_safety_rules(raw_cls, {
        "daily_sm_hours":       daily_sm_hours,
        "sleep_hours":          sleep_hours,
        "stress_experience":    stress_experience,
        "avg_mood":             avg_mood,
        "before_sleep_usage":   before_sleep_usage,
        "comparison_frequency": comparison_frequency,
        "productivity_impact":  productivity_impact,
    })

    rec = RECOMMENDATIONS[final_cls]

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.78rem;font-weight:700;letter-spacing:1.4px;text-transform:uppercase;color:#818cf8;margin-bottom:1rem;">'
        '📊 Analysis Results</p>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Your Mental Health Level</div>
            <div style="font-size:2rem; margin-bottom:0.3rem;">{rec['emoji']}</div>
            <span class="badge {rec['badge']}">{rec['label']}</span>
            <p style="color:#94a3b8; font-size:0.92rem; margin-top:1rem; line-height:1.6;">
                {rec['summary']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        if rule_warnings:
            for w in rule_warnings:
                st.markdown(f'<div class="risk-alert">{w}</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="card" style="margin-top:0.9rem;">
            <div class="card-title">Model Confidence</div>
        """, unsafe_allow_html=True)

        COLORS = ["#ef4444", "#f97316", "#f59e0b", "#22c55e", "#3b82f6"]
        for i, (cn, p) in enumerate(zip(class_names, proba)):
            is_pred     = (i == final_cls)
            fill_class  = "prob-fill" if is_pred else "prob-fill-dim"
            label_style = "color:#e2e8f0; font-weight:600;" if is_pred else ""
            pct_style   = f"color:{COLORS[i]}; font-weight:600;" if is_pred else "color:#475569;"
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-label" style="{label_style}">{cn}</div>
                <div class="prob-track">
                    <div class="{fill_class}" style="width:{p*100:.1f}%;
                        {'background:'+COLORS[i]+';' if is_pred else ''}"></div>
                </div>
                <div class="prob-pct" style="{pct_style}">{p*100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(f"""
        <div class="card" style="height:100%;">
            <div class="card-title">💡 Personalised Recommendations</div>
            <p style="color:#64748b; font-size:0.83rem; margin-bottom:0.8rem;">
                Based on your <strong style='color:#818cf8;'>{rec['label']}</strong> assessment
            </p>
        """, unsafe_allow_html=True)
        for r in rec["recs"]:
            st.markdown(f'<div class="rec">{r}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📝 View Your Input Summary"):
        pairs = [
            ("Age Group", age), ("Gender", gender), ("Occupation", occupation),
            ("Daily SM Usage", daily_sm_hours), ("Peak Usage Time", peak_usage_time),
            ("Before Sleep", before_sleep_usage), ("Purpose", purpose),
            ("Sleep Duration", sleep_hours), ("Stress Frequency", stress_experience),
            ("Average Mood", avg_mood), ("Productivity Impact", productivity_impact),
            ("Comparison Frequency", comparison_frequency),
        ]
        html = '<div style="padding:0.3rem 0;">'
        for k, v in pairs:
            html += f'<div class="sum-row"><span class="sum-key">{k}</span><span class="sum-val">{v}</span></div>'
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
