import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# TRUNA-ELE Navigator
# app.py demo textos v2
#
# Objetivo práctico:
# - El usuario final sube un Excel con columnas:
#     sujeto | texto
# - La app entrega nivel estimado, confianza, probabilidades,
#   perfil multidimensional e interpretación.
#
# Estado:
# - Texto/Excel con textos: motor estimativo interno.
# - Excel con índices TRUNAJOD: si existe motor externo disponible,
#   puede usar motor_narrativo_v2_defendible.py.
#
# Requisito opcional:
# carpetas/modelos/motor_narrativo_v2_defendible.py
# ============================================================

LEVELS = ["A1", "A2", "B1", "B2", "C1"]

DIMENSIONS_BY_MODE = {
    "Narrativo": [
        "Dispersión y Variedad Estructural",
        "Coherencia Semántica Global y Progresión",
        "Riqueza Léxica Nominal y Precisión",
        "Carga Emocional Positiva-Negativa",
        "Referencialidad Difusa y Conectividad Afectiva",
        "Centralidad Discursiva y Estabilidad",
    ],
    "Argumentativo": [
        "Cohesión Local y Diversidad Funcional",
        "Coherencia Global y Organización Semántica",
        "Riqueza Léxica y Precisión Conceptual",
        "Organización Argumentativa y Marcadores Discursivos",
        "Construcción Sintáctica y Organización Informativa",
        "Posicionamiento Discursivo y Polaridad",
    ],
}

CONNECTORS = {
    "y", "pero", "porque", "aunque", "entonces", "después", "luego", "también",
    "además", "sin embargo", "por eso", "por lo tanto", "cuando", "mientras",
    "antes", "finalmente", "primero", "segundo", "por ejemplo", "no obstante",
    "por consiguiente", "en consecuencia", "a pesar de", "por otra parte",
    "finalmente", "en primer lugar", "en segundo lugar"
}

POS_WORDS = {
    "feliz", "contento", "alegre", "bueno", "bonito", "interesante",
    "maravilloso", "mejor", "valioso", "agradable", "importante",
    "significativo", "positivo", "confianza", "aprendizaje"
}

NEG_WORDS = {
    "triste", "malo", "difícil", "problema", "peor", "aburrido",
    "cansado", "preocupado", "miedo", "ansiedad", "incertidumbre",
    "pérdida", "conflicto", "negativo", "riesgo"
}


# ============================================================
# Carga opcional de motor externo con índices TRUNAJOD
# ============================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "carpetas" / "modelos"

if MODEL_DIR.exists():
    sys.path.append(str(MODEL_DIR))

try:
    from motor_narrativo_v2_defendible import predict_narrative_v2, MODEL_PARAMS
    NARRATIVE_V2_AVAILABLE = True
except Exception:
    predict_narrative_v2 = None
    MODEL_PARAMS = None
    NARRATIVE_V2_AVAILABLE = False


def excel_has_narrative_v2_features(df):
    """Verifica si el Excel contiene las columnas requeridas por el motor externo."""
    if not NARRATIVE_V2_AVAILABLE or MODEL_PARAMS is None:
        return False
    required = MODEL_PARAMS.get("features", [])
    return all(col in df.columns for col in required)


# ============================================================
# Configuración página
# ============================================================

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


# ============================================================
# Utilidades texto
# ============================================================

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
    text = str(text)

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

    punct_ratio = len(re.findall(r"[,;:]", text)) / max(n_words, 1)
    paragraphs = max(len([p for p in text.split("\n") if p.strip()]), 1)
    lexical_density = sum(1 for x in w if len(x) > 4) / n_words if n_words else 0

    # Proxies interpretativas adicionales para hacer más sensible el demo
    abstract_ratio = sum(len(x) >= 10 for x in w) / n_words if n_words else 0
    subordinate_markers = {
        "aunque", "mientras", "cuando", "porque", "si", "como", "donde",
        "quien", "que", "cual", "mientras", "después"
    }
    subordinate_ratio = sum(1 for x in w if x in subordinate_markers) / n_words if n_words else 0

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
        "abstract_ratio": abstract_ratio,
        "subordinate_ratio": subordinate_ratio,
    }


# ============================================================
# Motores estimativos desde texto
# ============================================================

def dimension_scores_narrative(text):
    f = extract_text_features(text)

    weights = {
        "Dispersión y Variedad Estructural": 22.69,
        "Coherencia Semántica Global y Progresión": 18.29,
        "Riqueza Léxica Nominal y Precisión": 15.86,
        "Carga Emocional Positiva-Negativa": 12.01,
        "Referencialidad Difusa y Conectividad Afectiva": 9.70,
        "Centralidad Discursiva y Estabilidad": 8.65,
    }

    scores = {
        "Dispersión y Variedad Estructural": round(np.mean([
            pct(f["avg_sentence_length"], 4, 30),
            pct(f["paragraphs"], 1, 5),
            pct(f["punct_per_word"], 0.005, 0.16),
            pct(f["subordinate_ratio"], 0.00, 0.08),
        ]), 1),

        "Coherencia Semántica Global y Progresión": round(np.mean([
            pct(f["connector_ratio"], 0.003, 0.085),
            pct(f["n_sents"], 2, 18),
            pct(f["paragraphs"], 1, 5),
        ]), 1),

        "Riqueza Léxica Nominal y Precisión": round(np.mean([
            pct(f["ttr"], 0.22, 0.78),
            pct(f["lexical_density"], 0.18, 0.72),
            pct(f["long_word_ratio"], 0.01, 0.38),
            pct(f["avg_word_len"], 3.3, 6.8),
            pct(f["abstract_ratio"], 0.00, 0.12),
        ]), 1),

        "Carga Emocional Positiva-Negativa": round(np.mean([
            pct(f["pos_ratio"], 0.000, 0.040),
            pct(f["neg_ratio"], 0.000, 0.035),
        ]), 1),

        "Referencialidad Difusa y Conectividad Afectiva": round(np.mean([
            pct(f["n_words"], 30, 380),
            pct(f["connector_ratio"], 0.003, 0.085),
            pct(f["pos_ratio"] + f["neg_ratio"], 0.000, 0.060),
        ]), 1),
    }

    centralidad = np.mean([
        100 - abs(pct(f["avg_sentence_length"], 4, 30) - 58) * 1.1,
        100 - abs(pct(f["ttr"], 0.22, 0.78) - 58) * 1.1,
        100 - abs(pct(f["connector_ratio"], 0.003, 0.085) - 55) * 1.1,
    ])
    scores["Centralidad Discursiva y Estabilidad"] = round(float(np.clip(centralidad, 0, 100)), 1)

    total_weight = sum(weights.values())
    weighted_profile = sum(scores[k] * weights[k] for k in weights) / total_weight

    # Calibración textual demo v2.1:
    # Se agrega una señal de sofisticación textual para separar mejor extremos A1-C1
    # cuando el usuario sube solo sujeto + texto y TRUNAJOD completo aún no está integrado.
    proficiency_proxy = np.mean([
        pct(f["n_words"], 70, 220),
        pct(f["avg_sentence_length"], 7, 18),
        pct(f["avg_word_len"], 3.6, 4.6),
        pct(f["long_word_ratio"], 0.10, 0.25),
        pct(f["connector_ratio"], 0.03, 0.08),
        pct(f["abstract_ratio"], 0.01, 0.07),
    ])

    adjusted_profile = (0.35 * weighted_profile) + (0.65 * proficiency_proxy)

    scores["Puntaje demo textual ajustado"] = round(adjusted_profile, 1)
    scores["Perfil multidimensional %"] = round(adjusted_profile, 1)

    return scores


def dimension_scores_argumentative(text):
    f = extract_text_features(text)

    weights = {
        "Cohesión Local y Diversidad Funcional": 20,
        "Coherencia Global y Organización Semántica": 18,
        "Riqueza Léxica y Precisión Conceptual": 17,
        "Organización Argumentativa y Marcadores Discursivos": 17,
        "Construcción Sintáctica y Organización Informativa": 16,
        "Posicionamiento Discursivo y Polaridad": 12,
    }

    scores = {
        "Cohesión Local y Diversidad Funcional": round(np.mean([
            pct(f["connector_ratio"], 0.003, 0.09),
            pct(f["ttr"], 0.22, 0.78),
            pct(f["n_sents"], 2, 18),
        ]), 1),

        "Coherencia Global y Organización Semántica": round(np.mean([
            pct(f["paragraphs"], 1, 5),
            pct(f["avg_sentence_length"], 6, 32),
            pct(f["punct_per_word"], 0.005, 0.17),
        ]), 1),

        "Riqueza Léxica y Precisión Conceptual": round(np.mean([
            pct(f["lexical_density"], 0.18, 0.75),
            pct(f["avg_word_len"], 3.3, 7.0),
            pct(f["long_word_ratio"], 0.01, 0.40),
            pct(f["abstract_ratio"], 0.00, 0.14),
        ]), 1),

        "Organización Argumentativa y Marcadores Discursivos": round(np.mean([
            pct(f["connector_ratio"], 0.003, 0.10),
            pct(f["punct_per_word"], 0.005, 0.17),
            pct(f["avg_sentence_length"], 8, 34),
        ]), 1),

        "Construcción Sintáctica y Organización Informativa": round(np.mean([
            pct(f["avg_sentence_length"], 6, 34),
            pct(f["lexical_density"], 0.18, 0.75),
            pct(f["subordinate_ratio"], 0.00, 0.08),
        ]), 1),

        "Posicionamiento Discursivo y Polaridad": round(np.mean([
            pct(f["neg_ratio"], 0.000, 0.035),
            pct(f["pos_ratio"], 0.000, 0.040),
            pct(f["connector_ratio"], 0.003, 0.09),
        ]), 1),
    }

    total_weight = sum(weights.values())
    scores["Perfil multidimensional %"] = round(
        sum(scores[k] * weights[k] for k in weights) / total_weight, 1
    )

    return scores


def estimate_level(profile):
    score = profile["Perfil multidimensional %"]

    # Cortes demo v2.1:
    # Ajustados para evitar concentración excesiva en B1/B2 en archivos de textos.
    # Esta calibración es demostrativa; no reemplaza el modelo científico final con TRUNAJOD completo.
    if score < 24:
        level = "A1"
    elif score < 40:
        level = "A2"
    elif score < 55:
        level = "B1"
    elif score < 64:
        level = "B2"
    else:
        level = "C1"

    centers = {
        "A1": 14,
        "A2": 32,
        "B1": 48,
        "B2": 59,
        "C1": 70,
    }

    distances = np.array([abs(score - centers[l]) for l in LEVELS])
    raw = np.exp(-distances / 9)
    probs = raw / raw.sum()
    confidence = float(probs[LEVELS.index(level)] * 100)

    return level, round(confidence, 1), {
        f"Prob. {l}": round(float(p * 100), 1)
        for l, p in zip(LEVELS, probs)
    }


# ============================================================
# Análisis
# ============================================================

def analyze_text_dataframe(df, mode_label):
    if "texto" not in df.columns:
        raise ValueError("El archivo debe incluir una columna llamada 'texto'.")

    if "sujeto" not in df.columns:
        df.insert(0, "sujeto", [f"texto_{i+1}" for i in range(len(df))])

    rows = []

    for _, row in df.iterrows():
        text = str(row["texto"])

        if mode_label == "Narrativo":
            profile = dimension_scores_narrative(text)
        else:
            profile = dimension_scores_argumentative(text)

        level, conf, probs = estimate_level(profile)

        out = row.to_dict()
        out.update({
            "Tipo de tarea": mode_label,
            "motor_utilizado": "Motor textual estimativo TRUNA-ELE",
            "Nivel estimado": level,
            "Confianza": conf,
        })
        out.update(probs)
        out.update(profile)
        rows.append(out)

    return pd.DataFrame(rows)


def analyze_dataframe(df, mode_label, source_kind):
    """
    Prioridad:
    1. Si modo Narrativo + Excel con índices TRUNAJOD completos + motor externo disponible:
       usar Motor Narrativo v2 Defendible.
    2. En cualquier otro caso:
       usar Motor textual estimativo sobre columna texto.
    """
    if mode_label == "Narrativo" and source_kind == "excel" and excel_has_narrative_v2_features(df):
        result = predict_narrative_v2(df)
        result["Tipo de tarea"] = "Narrativo"
        result["Nivel estimado"] = result["nivel_estimado_v2"]
        result["Confianza"] = result["confianza_v2"]
        return result

    return analyze_text_dataframe(df, mode_label)


def to_excel_bytes(result, mode_label):
    dimensions = DIMENSIONS_BY_MODE[mode_label]
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        result.to_excel(writer, index=False, sheet_name="Resultados")

        if "Nivel estimado" in result.columns:
            resumen = result["Nivel estimado"].value_counts().rename_axis("Nivel").reset_index(name="n_textos")
            resumen.to_excel(writer, index=False, sheet_name="Resumen")

        base_cols = [
            "sujeto", "Tipo de tarea", "motor_utilizado",
            "Nivel estimado", "Confianza", "Perfil multidimensional %",
            "nivel_estimado_v2_num", "confianza_v2",
            "margen_decision_v2", "zona_transicion_v2"
        ]

        prob_cols = [
            c for c in result.columns
            if c.startswith("Prob.") or c.startswith("Prob_v2_")
        ]

        cols = [
            c for c in base_cols + prob_cols + dimensions
            if c in result.columns
        ]

        if cols:
            result[cols].to_excel(writer, index=False, sheet_name="Perfil")

    return buffer.getvalue()


def excel_example_bytes():
    example = pd.DataFrame({
        "sujeto": ["demo_001", "demo_002"],
        "texto": [
            "Ayer fui al parque con mi familia. Primero caminamos cerca del río y después comimos juntos. Fue un día muy bonito porque todos estábamos contentos.",
            "Uno de los acontecimientos que más ha influido en mi desarrollo personal ocurrió durante un intercambio académico realizado en otro país. Antes de viajar imaginaba que la experiencia consistiría principalmente en perfeccionar mis competencias lingüísticas y conocer una cultura diferente. Sin embargo, los aprendizajes más significativos surgieron de situaciones inesperadas relacionadas con la adaptación social."
        ]
    })

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        example.to_excel(writer, index=False, sheet_name="Textos")
    return buffer.getvalue()


# ============================================================
# Estado sesión: reset de uploader Excel
# ============================================================

if "excel_uploader_key" not in st.session_state:
    st.session_state["excel_uploader_key"] = 0


# ============================================================
# Interfaz
# ============================================================

st.sidebar.markdown("### Configuración")
mode_label = st.sidebar.radio("Tipo de tarea", ["Narrativo", "Argumentativo"])
st.sidebar.success(f"Modo seleccionado: {mode_label}")
st.sidebar.markdown("---")

st.sidebar.caption(
    "El usuario final puede subir un Excel con sujeto + texto. "
    "Los índices lingüísticos se estiman internamente para el demo."
)

st.markdown("""
<div class="hero">
    <h1>TRUNA-ELE Navigator</h1>
    <p>Posicionamiento lingüístico-discursivo automatizado para Español como Lengua Extranjera</p>
    <div class="badge-row">
        <span class="badge">A1–C1</span>
        <span class="badge">Narrativo / Argumentativo</span>
        <span class="badge">Perfil multidimensional</span>
        <span class="badge">Excel de textos</span>
    </div>
</div>
""", unsafe_allow_html=True)


with st.expander("Información metodológica", expanded=False):
    st.markdown("""
Esta versión permite analizar archivos Excel con columnas **sujeto** y **texto**, generando un nivel estimado y un perfil multidimensional.

Para efectos de demostración, el sistema utiliza un motor textual estimativo calibrado para separar mejor niveles extremos, inspirado en las dimensiones TRUNA-ELE. En versiones posteriores, esta ruta será reemplazada por la integración completa de TRUNAJOD detrás de la interfaz.

Si el archivo contiene índices TRUNAJOD completos y está disponible el motor externo, la aplicación puede usar el Motor Narrativo v2 Defendible.
""")


tab1, tab2 = st.tabs(["📄 Subir Excel", "✍️ Analizar un texto"])

with tab1:
    st.markdown('<div class="section-title">Analizar archivo Excel</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">El archivo debe contener al menos una columna <b>texto</b>. Se recomienda incluir también <b>sujeto</b>.</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Subir Excel",
        type=["xlsx"],
        label_visibility="collapsed",
        key=f"excel_uploader_{st.session_state['excel_uploader_key']}"
    )

    col_excel_a, col_excel_b = st.columns([1, 1])

    with col_excel_a:
        st.download_button(
            "Descargar plantilla de ejemplo",
            data=excel_example_bytes(),
            file_name="ejemplo_TRUNA_ELE_textos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with col_excel_b:
        if st.button("Nuevo análisis Excel", use_container_width=True):
            st.session_state["excel_uploader_key"] += 1
            st.rerun()

with tab2:
    st.markdown('<div class="section-title">Analizar producción individual</div>', unsafe_allow_html=True)

    single_id = st.text_input("ID del texto", value="demo_001")
    single_text = st.text_area(
        "Producción escrita",
        height=220,
        placeholder="Pegar aquí una producción narrativa o argumentativa..."
    )

    col_a, col_b = st.columns([1, 1])

    with col_a:
        run_single = st.button("Analizar texto", use_container_width=True)

    with col_b:
        clear_note = st.button("Nuevo análisis", use_container_width=True)
        if clear_note:
            st.info("Para un nuevo análisis, borra o reemplaza el texto anterior y vuelve a presionar Analizar.")


try:
    df = None
    source_kind = None

    if uploaded is not None:
        df = pd.read_excel(uploaded)
        source_kind = "excel"
    elif run_single:
        df = pd.DataFrame({
            "sujeto": [single_id],
            "texto": [single_text]
        })
        source_kind = "text"

    if df is not None:
        result = analyze_dataframe(df, mode_label, source_kind)
        dimensions = DIMENSIONS_BY_MODE[mode_label]

        st.markdown('<div class="section-title">Resultado principal</div>', unsafe_allow_html=True)

        motor_name = result["motor_utilizado"].iloc[0] if "motor_utilizado" in result.columns else "No especificado"
        st.info(f"Motor utilizado: {motor_name}")

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

                if "Perfil multidimensional %" in result.columns:
                    st.metric("Perfil multidimensional", f'{r["Perfil multidimensional %"]}%')

                if "margen_decision_v2" in result.columns:
                    st.metric("Margen decisión", f'{r["margen_decision_v2"]}%')

            with c3:
                prob_cols = [
                    c for c in result.columns
                    if c.startswith("Prob.") or c.startswith("Prob_v2_")
                ]

                cols_to_show = [
                    c for c in ["sujeto", "Tipo de tarea", "zona_transicion_v2"] + prob_cols
                    if c in result.columns
                ]
                st.dataframe(result[cols_to_show], use_container_width=True, hide_index=True)

            if "zona_transicion_v2" in result.columns:
                st.warning(f"Zona de transición: {r['zona_transicion_v2']}")

            available_dimensions = [c for c in dimensions if c in result.columns]
            if available_dimensions:
                st.markdown('<div class="section-title">Perfil multidimensional</div>', unsafe_allow_html=True)
                vals = result.iloc[0][available_dimensions].astype(float)
                st.bar_chart(vals)

                profile_cols = [
                    c for c in ["sujeto", "Tipo de tarea", "Nivel estimado"] + available_dimensions
                    if c in result.columns
                ]
                st.dataframe(result[profile_cols], use_container_width=True, hide_index=True)

        else:
            main_cols = [
                c for c in [
                    "sujeto", "Tipo de tarea", "motor_utilizado",
                    "Nivel estimado", "Confianza",
                    "margen_decision_v2", "zona_transicion_v2",
                    "Perfil multidimensional %"
                ]
                if c in result.columns
            ]

            prob_cols = [
                c for c in result.columns
                if c.startswith("Prob.") or c.startswith("Prob_v2_")
            ]

            st.dataframe(result[main_cols + prob_cols], use_container_width=True, hide_index=True)

            available_dimensions = [c for c in dimensions if c in result.columns]
            if available_dimensions:
                st.markdown('<div class="section-title">Perfil multidimensional</div>', unsafe_allow_html=True)
                profile_cols = [
                    c for c in ["sujeto", "Tipo de tarea", "Nivel estimado"] + available_dimensions
                    if c in result.columns
                ]
                st.dataframe(result[profile_cols], use_container_width=True, hide_index=True)

        st.download_button(
            "Descargar resultados en Excel",
            data=to_excel_bytes(result, mode_label),
            file_name=f"TRUNA_ELE_Navigator_{mode_label}_resultados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    else:
        st.info("Sube un archivo o pega un texto para iniciar la demostración.")

except Exception as exc:
    st.error(f"No fue posible procesar la entrada: {exc}")

