import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# TRUNA-ELE Navigator
# app.py integrado v2
#
# Modos:
# 1) Texto libre:
#    - Usa motor textual demo/heurístico.
#
# 2) Excel con índices TRUNAJOD:
#    - Si modo = Narrativo y el Excel contiene las columnas requeridas,
#      usa motor_narrativo_v2_defendible.py desde carpetas/modelos/.
#
# Requisito en GitHub:
# carpetas/modelos/motor_narrativo_v2_defendible.py
# carpetas/modelos/metadata_truna_ele_narrativo_v2_defendible.json
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
    "antes", "finalmente", "primero", "segundo", "por ejemplo"
}

POS_WORDS = {
    "feliz", "contento", "alegre", "bueno", "bonito",
    "interesante", "maravilloso", "mejor"
}

NEG_WORDS = {
    "triste", "malo", "difícil", "problema", "peor",
    "aburrido", "cansado", "preocupado"
}


# ============================================================
# Carga opcional del Motor Narrativo v2 Defendible
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
    """Verifica si el Excel contiene las columnas requeridas por el motor narrativo v2."""
    if not NARRATIVE_V2_AVAILABLE or MODEL_PARAMS is None:
        return False

    required = MODEL_PARAMS.get("features", [])
    return all(col in df.columns for col in required)


# ============================================================
# Configuración de página
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
# Funciones demo textuales
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
            pct(f["avg_sentence_length"], 5, 28),
            pct(f["paragraphs"], 1, 5),
            pct(f["punct_per_word"], 0.01, 0.15),
        ]), 1),

        "Coherencia Semántica Global y Progresión": round(np.mean([
            pct(f["connector_ratio"], 0.005, 0.08),
            pct(f["n_sents"], 2, 16),
            pct(f["avg_sentence_length"], 6, 24),
        ]), 1),

        "Riqueza Léxica Nominal y Precisión": round(np.mean([
            pct(f["ttr"], 0.25, 0.80),
            pct(f["lexical_density"], 0.20, 0.70),
            pct(f["long_word_ratio"], 0.02, 0.35),
            pct(f["avg_word_len"], 3.5, 6.5),
        ]), 1),

        "Carga Emocional Positiva-Negativa": round(np.mean([
            pct(f["pos_ratio"], 0.000, 0.040),
            pct(f["neg_ratio"], 0.000, 0.030),
        ]), 1),

        "Referencialidad Difusa y Conectividad Afectiva": round(np.mean([
            pct(f["n_words"], 40, 350),
            pct(f["connector_ratio"], 0.005, 0.08),
            pct(f["pos_ratio"] + f["neg_ratio"], 0.000, 0.050),
        ]), 1),
    }

    centralidad = np.mean([
        100 - abs(pct(f["avg_sentence_length"], 5, 28) - 55) * 1.2,
        100 - abs(pct(f["ttr"], 0.25, 0.80) - 55) * 1.2,
        100 - abs(pct(f["connector_ratio"], 0.005, 0.08) - 55) * 1.2,
    ])

    scores["Centralidad Discursiva y Estabilidad"] = round(float(np.clip(centralidad, 0, 100)), 1)

    total_weight = sum(weights.values())
    scores["Perfil multidimensional %"] = round(
        sum(scores[k] * weights[k] for k in weights) / total_weight, 1
    )

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
            pct(f["connector_ratio"], 0.005, 0.08),
            pct(f["ttr"], 0.25, 0.80),
            pct(f["n_sents"], 2, 16),
        ]), 1),

        "Coherencia Global y Organización Semántica": round(np.mean([
            pct(f["paragraphs"], 1, 5),
            pct(f["avg_sentence_length"], 6, 28),
            pct(f["punct_per_word"], 0.01, 0.15),
        ]), 1),

        "Riqueza Léxica y Precisión Conceptual": round(np.mean([
            pct(f["lexical_density"], 0.20, 0.75),
            pct(f["avg_word_len"], 3.5, 6.8),
            pct(f["long_word_ratio"], 0.02, 0.38),
            pct(f["ttr"], 0.25, 0.80),
        ]), 1),

        "Organización Argumentativa y Marcadores Discursivos": round(np.mean([
            pct(f["connector_ratio"], 0.005, 0.10),
            pct(f["punct_per_word"], 0.01, 0.16),
            pct(f["avg_sentence_length"], 8, 32),
        ]), 1),

        "Construcción Sintáctica y Organización Informativa": round(np.mean([
            pct(f["avg_sentence_length"], 6, 32),
            pct(f["lexical_density"], 0.20, 0.75),
            pct(f["long_word_ratio"], 0.02, 0.38),
        ]), 1),

        "Posicionamiento Discursivo y Polaridad": round(np.mean([
            pct(f["neg_ratio"], 0.000, 0.035),
            pct(f["pos_ratio"], 0.000, 0.040),
            pct(f["connector_ratio"], 0.005, 0.08),
        ]), 1),
    }

    total_weight = sum(weights.values())
    scores["Perfil multidimensional %"] = round(
        sum(scores[k] * weights[k] for k in weights) / total_weight, 1
    )

    return scores


def estimate_level(profile):
    score = profile["Perfil multidimensional %"]

    if score < 18:
        level = "A1"
    elif score < 32:
        level = "A2"
    elif score < 48:
        level = "B1"
    elif score < 62:
        level = "B2"
    else:
        level = "C1"

    centers = {
        "A1": 10,
        "A2": 25,
        "B1": 40,
        "B2": 55,
        "C1": 72,
    }

    distances = np.array([abs(score - centers[l]) for l in LEVELS])
    raw = np.exp(-distances / 10)
    probs = raw / raw.sum()
    confidence = float(probs[LEVELS.index(level)] * 100)

    return level, round(confidence, 1), {
        f"Prob. {l}": round(float(p * 100), 1)
        for l, p in zip(LEVELS, probs)
    }


# ============================================================
# Análisis
# ============================================================

def analyze_text_demo_dataframe(df, mode_label):
    """Ruta demo: analiza texto crudo con heurísticas simples."""
    if "texto" not in df.columns:
        raise ValueError("El archivo debe incluir una columna llamada 'texto' para usar la ruta demo textual.")

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
            "motor_utilizado": "Demo textual heurística",
            "Nivel estimado": level,
            "Confianza": conf,
        })
        out.update(probs)
        out.update(profile)
        rows.append(out)

    return pd.DataFrame(rows)


def analyze_dataframe(df, mode_label, source_kind):
    """
    Decide qué motor usar.

    source_kind:
    - "excel": archivo subido.
    - "text": texto pegado.
    """
    if mode_label == "Narrativo" and source_kind == "excel" and excel_has_narrative_v2_features(df):
        result = predict_narrative_v2(df)

        # Normalización de nombres para que la interfaz use columnas comunes.
        result["Tipo de tarea"] = "Narrativo"
        result["Nivel estimado"] = result["nivel_estimado_v2"]
        result["Confianza"] = result["confianza_v2"]

        return result

    return analyze_text_demo_dataframe(df, mode_label)


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
        "sujeto": ["demo_narrativo_001", "demo_argumentativo_001"],
        "texto": [
            "Ayer fui al parque con mi familia. Primero caminamos cerca del río y después comimos juntos. Fue un día muy bonito porque todos estábamos contentos.",
            "La inteligencia artificial puede ayudar a los estudiantes, pero también exige responsabilidad. Por eso, es importante aprender a usarla de manera crítica y comprender sus límites."
        ]
    })

    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        example.to_excel(writer, index=False, sheet_name="Textos")

    return buffer.getvalue()


# ============================================================
# Interfaz
# ============================================================

st.sidebar.markdown("### Configuración")
mode_label = st.sidebar.radio("Tipo de tarea", ["Narrativo", "Argumentativo"])
st.sidebar.success(f"Modo seleccionado: {mode_label}")
st.sidebar.markdown("---")

if mode_label == "Narrativo":
    if NARRATIVE_V2_AVAILABLE:
        st.sidebar.caption(
            "Motor Narrativo v2 disponible para Excel con índices TRUNAJOD. "
            "Texto libre usa demo heurística."
        )
    else:
        st.sidebar.warning(
            "Motor Narrativo v2 no detectado. "
            "Verifica que exista carpetas/modelos/motor_narrativo_v2_defendible.py"
        )
else:
    st.sidebar.caption(
        "Motor Argumentativo v0.1: versión exploratoria preliminar basada en dimensiones argumentativas en desarrollo."
    )


st.markdown("""
<div class="hero">
    <h1>TRUNA-ELE Navigator</h1>
    <p>Posicionamiento lingüístico-discursivo automatizado para Español como Lengua Extranjera</p>
    <div class="badge-row">
        <span class="badge">A1–C1</span>
        <span class="badge">Narrativo / Argumentativo</span>
        <span class="badge">Perfil multidimensional</span>
        <span class="badge">Motor TRUNAJOD para Excel</span>
    </div>
</div>
""", unsafe_allow_html=True)


with st.expander("Información metodológica", expanded=False):
    if mode_label == "Narrativo":
        st.markdown("""
Esta versión incorpora dos rutas de análisis:

**1. Texto libre:** usa una ruta demo heurística que permite probar el flujo general de la plataforma.

**2. Excel con índices TRUNAJOD:** si el archivo contiene los índices requeridos por el Motor Narrativo v2 Defendible, el sistema utiliza un modelo transparente basado en índices TRUNAJOD reales.

El Motor Narrativo v2 Defendible se concibe como un sistema de **posicionamiento lingüístico-discursivo**, no como una reproducción mecánica de etiquetas humanas. Dado el solapamiento observado entre niveles humanos, el sistema reporta nivel estimado, confianza, margen de decisión y zona de transición.
""")
    else:
        st.markdown("""
Esta versión utiliza un **Motor Argumentativo v0.1**, construido como aproximación exploratoria inicial.

El perfil se organiza en seis dimensiones:

1. Cohesión Local y Diversidad Funcional.
2. Coherencia Global y Organización Semántica.
3. Riqueza Léxica y Precisión Conceptual.
4. Organización Argumentativa y Marcadores Discursivos.
5. Construcción Sintáctica y Organización Informativa.
6. Posicionamiento Discursivo y Polaridad.

Esta versión deberá ser refinada con el ACP formal argumentativo, la matriz de cargas y los modelos predictivos finales. Por ahora funciona como motor exploratorio de demostración.
""")


tab1, tab2 = st.tabs(["📄 Subir Excel", "✍️ Analizar un texto"])

with tab1:
    st.markdown('<div class="section-title">Analizar un archivo Excel</div>', unsafe_allow_html=True)

    if mode_label == "Narrativo":
        st.markdown(
            '<div class="muted">Si el Excel contiene índices TRUNAJOD narrativos, se usará el Motor Narrativo v2 Defendible. Si solo contiene <b>texto</b>, se usará la ruta demo textual.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="muted">El archivo debe contener al menos una columna <b>texto</b>. Opcionalmente puede incluir <b>sujeto</b>.</div>',
            unsafe_allow_html=True
        )

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

            # Perfil gráfico: solo disponible en ruta demo textual.
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

