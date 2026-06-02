
import io
import re
import numpy as np
import pandas as pd
import streamlit as st

LEVELS = ["A1", "A2", "B1", "B2", "C1"]

DIMENSIONS = [
    "Dispersión y variedad estructural",
    "Coherencia semántica global",
    "Riqueza léxica y precisión",
    "Carga emocional",
    "Referencialidad y conectividad",
    "Centralidad discursiva y estabilidad",
]

CONNECTORS = {
    "y", "pero", "porque", "aunque", "entonces", "después", "luego", "también",
    "además", "sin embargo", "por eso", "por lo tanto", "cuando", "mientras",
    "antes", "finalmente", "primero", "segundo", "por ejemplo"
}
POS_WORDS = {"feliz", "contento", "alegre", "bueno", "bonito", "interesante", "maravilloso", "mejor"}
NEG_WORDS = {"triste", "malo", "difícil", "problema", "peor", "aburrido", "cansado", "preocupado"}

st.set_page_config(
    page_title="TRUNA-ELE Navigator",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1180px;
}
.hero {
    padding: 2rem 2.2rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #0f2a5f 0%, #164a9f 55%, #5b2bbf 100%);
    color: white;
    box-shadow: 0 16px 40px rgba(15,42,95,.22);
    margin-bottom: 1.4rem;
}
.hero h1 {
    font-size: 3rem;
    margin: 0;
    letter-spacing: -1px;
}
.hero p {
    font-size: 1.15rem;
    opacity: .92;
    margin-top: .55rem;
    margin-bottom: 0;
}
.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: .6rem;
    margin-top: 1rem;
}
.badge {
    padding: .42rem .8rem;
    border-radius: 999px;
    background: rgba(255,255,255,.16);
    border: 1px solid rgba(255,255,255,.24);
    font-size: .88rem;
}
.metric-card {
    border-radius: 18px;
    padding: 1.2rem;
    background: linear-gradient(180deg, #ffffff 0%, #f7f9ff 100%);
    border: 1px solid #e4ebfa;
    text-align: center;
}
.level-big {
    font-size: 4rem;
    font-weight: 800;
    color: #164a9f;
    line-height: 1;
}
.muted {
    color: #667085;
    font-size: .95rem;
}
.section-title {
    font-size: 1.35rem;
    font-weight: 750;
    color: #172033;
    margin-top: 1rem;
    margin-bottom: .6rem;
}
.stButton>button {
    border-radius: 12px;
    font-weight: 700;
}
.stDownloadButton>button {
    border-radius: 12px;
    font-weight: 700;
}
[data-testid="stSidebar"] {
    background: #f2f6ff;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def words(text):
    return re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", str(text).lower())

def sentences(text):
    parts = re.split(r"[.!?¿¡]+", str(text))
    return [p.strip() for p in parts if p.strip()]

def pct(x, low, high):
    if pd.isna(x):
        return 50.0
    if high == low:
        return 50.0
    return float(np.clip((x - low) / (high - low) * 100, 0, 100))

def extract_text_features(text):
    w = words(text)
    s = sentences(text)
    n_words = len(w)
    n_unique = len(set(w))
    n_sents = max(len(s), 1)
    avg_word_len = np.mean([len(x) for x in w]) if w else 0
    avg_sent_len = n_words / n_sents if n_sents else 0
    ttr = n_unique / n_words if n_words else 0
    long_ratio = sum(len(x) >= 7 for x in w) / n_words if n_words else 0
    connector_count = sum(1 for x in w if x in CONNECTORS)
    connector_ratio = connector_count / n_words if n_words else 0
    pos_ratio = sum(1 for x in w if x in POS_WORDS) / n_words if n_words else 0
    neg_ratio = sum(1 for x in w if x in NEG_WORDS) / n_words if n_words else 0
    punct_ratio = len(re.findall(r"[,;:]", str(text))) / max(n_words, 1)
    paragraphs = max(len([p for p in str(text).split("\n") if p.strip()]), 1)
    lexical_density = sum(1 for x in w if len(x) > 4) / n_words if n_words else 0

    return {
        "n_words": n_words,
        "n_sents": n_sents,
        "avg_word_len": avg_word_len,
        "avg_sentence_length": avg_sent_len,
        "ttr": ttr,
        "long_word_ratio": long_ratio,
        "connector_ratio": connector_ratio,
        "pos_ratio": pos_ratio,
        "neg_ratio": neg_ratio,
        "punct_per_word": punct_ratio,
        "paragraphs": paragraphs,
        "lexical_density": lexical_density,
    }

def dimension_scores_from_text(text):
    f = extract_text_features(text)
    dispersion = np.mean([pct(f["avg_sentence_length"], 5, 28), pct(f["ttr"], 0.25, 0.80), pct(f["paragraphs"], 1, 4)])
    coherencia = np.mean([pct(f["connector_ratio"], 0.005, 0.08), pct(f["punct_per_word"], 0.01, 0.15), pct(f["n_sents"], 2, 14)])
    riqueza = np.mean([pct(f["lexical_density"], 0.20, 0.70), pct(f["avg_word_len"], 3.5, 6.5), pct(f["long_word_ratio"], 0.02, 0.35)])
    emocional = np.mean([pct(f["pos_ratio"], 0, 0.04), pct(f["neg_ratio"], 0, 0.03)])
    referencialidad = np.mean([pct(f["n_words"], 40, 350), pct(f["connector_ratio"], 0.005, 0.08)])
    centralidad = np.mean([100 - abs(pct(f["avg_sentence_length"], 5, 28) - 55) * 1.2, 100 - abs(pct(f["ttr"], 0.25, 0.80) - 55) * 1.2])
    centralidad = float(np.clip(centralidad, 0, 100))

    scores = {
        "Dispersión y variedad estructural": round(dispersion, 1),
        "Coherencia semántica global": round(coherencia, 1),
        "Riqueza léxica y precisión": round(riqueza, 1),
        "Carga emocional": round(emocional, 1),
        "Referencialidad y conectividad": round(referencialidad, 1),
        "Centralidad discursiva y estabilidad": round(centralidad, 1),
    }
    scores["Perfil multidimensional %"] = round(np.mean(list(scores.values())), 1)
    return scores

def estimate_level(profile):
    score = profile["Perfil multidimensional %"]
    if score < 25:
        level = "A1"
    elif score < 42:
        level = "A2"
    elif score < 60:
        level = "B1"
    elif score < 78:
        level = "B2"
    else:
        level = "C1"

    centers = {"A1": 15, "A2": 33, "B1": 51, "B2": 69, "C1": 87}
    distances = np.array([abs(score - centers[l]) for l in LEVELS])
    raw = np.exp(-distances / 12)
    probs = raw / raw.sum()
    confidence = float(probs[LEVELS.index(level)] * 100)
    return level, round(confidence, 1), {f"Prob. {l}": round(float(p * 100), 1) for l, p in zip(LEVELS, probs)}

def analyze_dataframe(df):
    if "texto" not in df.columns:
        raise ValueError("El archivo debe incluir una columna llamada 'texto'.")
    if "sujeto" not in df.columns:
        df.insert(0, "sujeto", [f"texto_{i+1}" for i in range(len(df))])

    rows = []
    for _, row in df.iterrows():
        text = str(row["texto"])
        profile = dimension_scores_from_text(text)
        level, conf, probs = estimate_level(profile)
        out = row.to_dict()
        out.update({"Nivel estimado": level, "Confianza": conf})
        out.update(probs)
        out.update(profile)
        rows.append(out)
    return pd.DataFrame(rows)

def to_excel_bytes(result):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        result.to_excel(writer, index=False, sheet_name="Resultados")
        resumen = result["Nivel estimado"].value_counts().rename_axis("Nivel").reset_index(name="n_textos")
        resumen.to_excel(writer, index=False, sheet_name="Resumen")
        cols = [c for c in ["sujeto", "Nivel estimado", "Confianza", "Perfil multidimensional %"] + DIMENSIONS if c in result.columns]
        result[cols].to_excel(writer, index=False, sheet_name="Perfil")
    return buffer.getvalue()

def excel_example_bytes():
    example = pd.DataFrame({
        "sujeto": ["demo_001", "demo_002"],
        "texto": [
            "Ayer fui al parque con mi familia. Primero caminamos cerca del río y después comimos juntos. Fue un día muy bonito porque todos estábamos contentos.",
            "La inteligencia artificial puede ayudar a los estudiantes, pero también exige responsabilidad. Por eso, es importante aprender a usarla de manera crítica."
        ]
    })
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        example.to_excel(writer, index=False, sheet_name="Textos")
    return buffer.getvalue()

st.sidebar.markdown("### Configuración")
mode_label = st.sidebar.radio("Tipo de tarea", ["Narrativo", "Argumentativo"])
st.sidebar.success(f"Modo seleccionado: {mode_label}")
st.sidebar.markdown("---")
st.sidebar.caption("Versión demo. El selector prepara la interfaz para futuros motores diferenciados por tarea.")

st.markdown("""
<div class="hero">
    <h1>TRUNA-ELE Navigator</h1>
    <p>Posicionamiento lingüístico-discursivo automatizado para Español como Lengua Extranjera</p>
    <div class="badge-row">
        <span class="badge">A1–C1</span>
        <span class="badge">Narrativo / Argumentativo</span>
        <span class="badge">Perfil multidimensional</span>
        <span class="badge">Demo funcional</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("Información metodológica", expanded=False):
    st.markdown("""
Esta versión permite demostrar el flujo general de TRUNA-ELE Navigator: **texto → análisis lingüístico → perfil multidimensional → nivel estimado**.

La versión actual utiliza un motor simplificado para demostración. La siguiente etapa consiste en integrar los índices predictivos específicos para tareas narrativas y argumentativas derivados de los estudios TRUNA-ELE.
""")

tab1, tab2 = st.tabs(["📄 Subir Excel", "✍️ Analizar un texto"])

with tab1:
    st.markdown('<div class="section-title">Analizar un archivo Excel</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">El archivo debe contener al menos una columna <b>texto</b>. Opcionalmente puede incluir <b>sujeto</b>.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Subir Excel", type=["xlsx"], label_visibility="collapsed")
    st.download_button(
        "Descargar plantilla de ejemplo",
        data=excel_example_bytes(),
        file_name="ejemplo_TRUNA_ELE_textos.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with tab2:
    st.markdown('<div class="section-title">Analizar una producción individual</div>', unsafe_allow_html=True)
    single_id = st.text_input("ID del texto", value="demo_001")
    single_text = st.text_area("Producción escrita", height=220, placeholder="Pegar aquí una producción narrativa o argumentativa...")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        run_single = st.button("Analizar texto", use_container_width=True)
    with col_b:
        clear_note = st.button("Nuevo análisis", use_container_width=True)
        if clear_note:
            st.info("Para un nuevo análisis, borra o reemplaza el texto anterior y vuelve a presionar Analizar.")

try:
    df = None
    if uploaded is not None:
        df = pd.read_excel(uploaded)
    elif run_single:
        df = pd.DataFrame({"sujeto": [single_id], "texto": [single_text]})

    if df is not None:
        result = analyze_dataframe(df)
        st.markdown('<div class="section-title">Resultado principal</div>', unsafe_allow_html=True)

        if len(result) == 1:
            r = result.iloc[0]
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="muted">Nivel estimado</div>
                    <div class="level-big">{r["Nivel estimado"]}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.metric("Confianza", f'{r["Confianza"]}%')
                st.metric("Perfil multidimensional", f'{r["Perfil multidimensional %"]}%')
            with c3:
                prob_cols = [c for c in result.columns if c.startswith("Prob.")]
                st.dataframe(result[["sujeto"] + prob_cols], use_container_width=True, hide_index=True)

            st.markdown('<div class="section-title">Perfil multidimensional</div>', unsafe_allow_html=True)
            vals = result.iloc[0][DIMENSIONS].astype(float)
            st.bar_chart(vals)

            profile_cols = [c for c in ["sujeto", "Nivel estimado"] + DIMENSIONS if c in result.columns]
            st.dataframe(result[profile_cols], use_container_width=True, hide_index=True)

        else:
            main_cols = [c for c in ["sujeto", "Nivel estimado", "Confianza", "Perfil multidimensional %"] if c in result.columns]
            prob_cols = [c for c in result.columns if c.startswith("Prob.")]
            st.dataframe(result[main_cols + prob_cols], use_container_width=True, hide_index=True)

            st.markdown('<div class="section-title">Perfil multidimensional</div>', unsafe_allow_html=True)
            profile_cols = [c for c in ["sujeto", "Nivel estimado"] + DIMENSIONS if c in result.columns]
            st.dataframe(result[profile_cols], use_container_width=True, hide_index=True)

        st.download_button(
            "Descargar resultados en Excel",
            data=to_excel_bytes(result),
            file_name="TRUNA_ELE_Navigator_resultados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("Sube un archivo o pega un texto para iniciar la demostración.")

except Exception as exc:
    st.error(f"No fue posible procesar la entrada: {exc}")


