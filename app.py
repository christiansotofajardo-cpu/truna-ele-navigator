import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from text_features import TextFeatureExtractor

APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / 'models'
EXAMPLES_DIR = APP_DIR / 'examples'
IMG_PATH = APP_DIR / 'assets' / 'truna_ele_infografia.png'

LEVEL_MAP = {1: 'A1', 2: 'A2', 3: 'B1', 4: 'B2', 5: 'C1'}
MODE_LABELS = {'Narrativo': 'narrativo', 'Argumentativo': 'argumentativo'}
DIMENSIONS = [
    'Dispersión y variedad estructural',
    'Coherencia semántica global',
    'Riqueza léxica y precisión',
    'Carga emocional',
    'Referencialidad y conectividad',
    'Centralidad discursiva y estabilidad',
]

@st.cache_resource
def load_metadata():
    return json.loads((MODELS_DIR / 'metadata.json').read_text(encoding='utf-8'))

@st.cache_resource
def load_model(kind: str, mode_key: str):
    return joblib.load(MODELS_DIR / f'modelo_{kind}_{mode_key}.joblib')


def pct_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    if s.notna().sum() <= 1 or s.max() == s.min():
        return pd.Series(50.0, index=s.index)
    return ((s - s.min()) / (s.max() - s.min()) * 100).clip(0, 100).round(1)


def safe_col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors='coerce')
    return pd.Series(default, index=df.index, dtype='float64')


def text_basic_features(texts: pd.Series) -> pd.DataFrame:
    extractor = TextFeatureExtractor()
    arr = extractor.transform(texts.fillna('').astype(str))
    return pd.DataFrame(arr, columns=extractor.feature_names, index=texts.index)


def dimensions_from_text(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    f = text_basic_features(out['texto'])
    raw = pd.DataFrame(index=out.index)
    raw['Dispersión y variedad estructural'] = pd.concat([
        pct_series(f['avg_sentence_length']), pct_series(f['ttr']), pct_series(f['paragraphs'])
    ], axis=1).mean(axis=1)
    raw['Coherencia semántica global'] = pd.concat([
        pct_series(f['connector_ratio']), pct_series(f['punct_per_word']), pct_series(f['n_sents'])
    ], axis=1).mean(axis=1)
    raw['Riqueza léxica y precisión'] = pd.concat([
        pct_series(f['lexical_density']), pct_series(f['avg_word_len']), pct_series(f['long_word_ratio'])
    ], axis=1).mean(axis=1)
    raw['Carga emocional'] = pd.concat([
        pct_series(f['pos_ratio']), pct_series(f['neg_ratio'])
    ], axis=1).mean(axis=1)
    raw['Referencialidad y conectividad'] = pd.concat([
        pct_series(f['n_words']), pct_series(f['connector_ratio'])
    ], axis=1).mean(axis=1)
    raw['Centralidad discursiva y estabilidad'] = pd.concat([
        100 - abs(pct_series(f['avg_sentence_length']) - 50) * 2,
        100 - abs(pct_series(f['ttr']) - 50) * 2,
    ], axis=1).mean(axis=1).clip(0, 100)
    for c in DIMENSIONS:
        out[c] = raw[c].round(1)
    out['Perfil multidimensional %'] = raw[DIMENSIONS].mean(axis=1).round(1)
    return out


def dimensions_from_indices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ttr = safe_col(out, 'TTR lemma')
    ttr_fun = safe_col(out, 'TTR function')
    mtld = safe_col(out, 'TTR Diversidad léxica MTLD')
    sent_len = safe_col(out, 'SP Promedio Longitud Oracion')
    lex_density = safe_col(out, 'SP Densidad Léxica')
    coh = safe_col(out, 'EG local_coherence_PACC')
    pu = safe_col(out, 'EG local_coherence_PU')
    pw = safe_col(out, 'EG local_coherence_PW')
    cos_dist = safe_col(out, 'avg_cos_dist')
    synt = safe_col(out, 'SP syntactic similarity')
    noun = safe_col(out, 'TTR noun')
    arg = safe_col(out, 'TTR argument')
    joy = safe_col(out, 'joy')
    pos = safe_col(out, 'POS')
    neg = safe_col(out, 'NEG')
    givenness = safe_col(out, 'avg_givenness')
    centroid = safe_col(out, 'avg_dist_to_centroid')
    dm = safe_col(out, 'DM all types of discourse markers')

    raw = pd.DataFrame(index=out.index)
    raw['Dispersión y variedad estructural'] = pd.concat([pct_series(ttr), pct_series(sent_len), pct_series(mtld)], axis=1).mean(axis=1)
    raw['Coherencia semántica global'] = pd.concat([pct_series(coh), pct_series(pu), pct_series(pw), 100 - pct_series(cos_dist)], axis=1).mean(axis=1)
    raw['Riqueza léxica y precisión'] = pd.concat([pct_series(lex_density), pct_series(noun), pct_series(arg), pct_series(mtld)], axis=1).mean(axis=1)
    raw['Carga emocional'] = pd.concat([pct_series(joy), pct_series(pos), pct_series(neg)], axis=1).mean(axis=1)
    raw['Referencialidad y conectividad'] = pd.concat([pct_series(givenness), 100 - pct_series(centroid), pct_series(dm)], axis=1).mean(axis=1)
    raw['Centralidad discursiva y estabilidad'] = pd.concat([pct_series(synt), pct_series(ttr_fun)], axis=1).mean(axis=1)
    for c in DIMENSIONS:
        out[c] = raw[c].round(1)
    out['Perfil multidimensional %'] = raw[DIMENSIONS].mean(axis=1).round(1)
    return out


def predict_with_model(df: pd.DataFrame, model, input_kind: str, metadata: dict) -> pd.DataFrame:
    if input_kind == 'texto':
        X = df['texto'].fillna('').astype(str)
        out = dimensions_from_text(df)
    else:
        features = metadata['features_indices']
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError('Faltan columnas TRUNAJOD para el modo índices: ' + ', '.join(missing[:15]))
        X = df[features]
        out = dimensions_from_indices(df)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    classes = model.named_steps['clf'].classes_
    out['Nivel_estimado_num'] = preds
    out['Nivel_estimado_MCER'] = [LEVEL_MAP.get(int(p), str(p)) for p in preds]
    out['Confianza_%'] = (probs.max(axis=1) * 100).round(1)
    for i, cls in enumerate(classes):
        out[f'Prob_{LEVEL_MAP.get(int(cls), str(cls))}_%'] = (probs[:, i] * 100).round(1)
    return out


def to_excel_bytes(result: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        result.to_excel(writer, index=False, sheet_name='Resultados')
        resumen = result['Nivel_estimado_MCER'].value_counts().rename_axis('Nivel').reset_index(name='n_textos')
        resumen.to_excel(writer, index=False, sheet_name='Resumen')
        result[[c for c in ['sujeto','Nivel_estimado_MCER','Confianza_%','Perfil multidimensional %'] + DIMENSIONS if c in result.columns]].to_excel(writer, index=False, sheet_name='Perfil')
    return buffer.getvalue()

st.set_page_config(page_title='TRUNA-ELE Navigator', page_icon='📘', layout='wide')
st.title('TRUNA-ELE Navigator v1.5')
st.caption('Sistema de posicionamiento lingüístico-discursivo para producciones escritas en ELE')

metadata = load_metadata()

with st.expander('Descripción breve', expanded=True):
    st.markdown('''
**TRUNA-ELE Navigator v1.5** estima el posicionamiento de producciones escritas de aprendientes de Español como Lengua Extranjera dentro del continuo **A1-C1**.  
La salida combina una **estimación de nivel** con un **perfil multidimensional** inspirado en las dimensiones identificadas en los trabajos TRUNA-ELE.

Esta versión incluye dos modos:

1. **Texto directo / Excel con textos**: permite una demo inmediata desde columnas `sujeto` y `texto`. Usa un extractor textual liviano incorporado en la app.
2. **Excel con índices TRUNAJOD**: usa el motor entrenado sobre índices TRUNAJOD cuando el archivo ya viene procesado.

La versión 1.5 debe presentarse como **prototipo demostrativo y exploratorio**, no como sustituto de evaluación experta ni como versión final validada externamente.
''')

mode_label = st.sidebar.radio('Tipo de tarea', list(MODE_LABELS.keys()))
mode_key = MODE_LABELS[mode_label]
input_label = st.sidebar.radio('Modo de entrada', ['Texto directo / Excel con textos', 'Excel con índices TRUNAJOD'])
input_kind = 'texto' if input_label.startswith('Texto') else 'indices'
model = load_model(input_kind, mode_key)
info = metadata['tasks'][mode_key][f'{input_kind}_model']

st.sidebar.subheader('Motor cargado')
st.sidebar.write(f'**Tarea:** {mode_label}')
st.sidebar.write(f'**Entrada:** {input_label}')
st.sidebar.write(f'**Corpus entrenamiento:** {metadata["tasks"][mode_key]["n"]} textos')
st.sidebar.caption(f'Métrica holdout interna: accuracy={info["holdout_accuracy"]}, F1 macro={info["holdout_f1_macro"]}')

if IMG_PATH.exists():
    with st.expander('Imagen conceptual TRUNA-ELE', expanded=False):
        st.image(str(IMG_PATH), use_container_width=True)

st.subheader('Entrada')
if input_kind == 'texto':
    tab1, tab2 = st.tabs(['Subir Excel', 'Analizar un texto'])
    with tab1:
        uploaded = st.file_uploader('Subir Excel con columnas sujeto y texto', type=['xlsx'], key='excel_texto')
        ex = EXAMPLES_DIR / f'ejemplo_textos_{mode_key}.xlsx'
        if ex.exists():
            st.download_button('Descargar ejemplo de textos', data=ex.read_bytes(), file_name=ex.name)
        run_excel = uploaded is not None
    with tab2:
        single_id = st.text_input('ID del texto', value='demo_001')
        single_text = st.text_area('Pegar producción escrita', height=220)
        run_single = st.button('Analizar texto pegado')
else:
    uploaded = st.file_uploader('Subir Excel con índices TRUNAJOD', type=['xlsx'], key='excel_indices')
    ex = EXAMPLES_DIR / f'ejemplo_indices_{mode_key}.xlsx'
    if ex.exists():
        st.download_button('Descargar ejemplo con índices TRUNAJOD', data=ex.read_bytes(), file_name=ex.name)
    run_excel = uploaded is not None
    run_single = False

try:
    df = None
    if input_kind == 'texto' and run_single:
        df = pd.DataFrame({'sujeto': [single_id], 'texto': [single_text]})
    elif run_excel:
        df = pd.read_excel(uploaded)
    if df is not None:
        if input_kind == 'texto':
            if 'texto' not in df.columns:
                st.error('El archivo debe incluir una columna llamada texto.')
                st.stop()
            if 'sujeto' not in df.columns:
                df.insert(0, 'sujeto', [f'texto_{i+1}' for i in range(len(df))])
        else:
            if 'tarea' in df.columns:
                df = df[df['tarea'] == metadata['tasks'][mode_key]['tarea']].copy()
                if df.empty:
                    st.error('No hay filas para la tarea seleccionada según la columna tarea.')
                    st.stop()
        result = predict_with_model(df, model, input_kind, metadata)
        st.success(f'Análisis completado: {len(result)} texto(s).')

        main_cols = [c for c in ['sujeto','Nivel_estimado_MCER','Confianza_%','Perfil multidimensional %'] if c in result.columns]
        prob_cols = [c for c in result.columns if c.startswith('Prob_')]
        st.subheader('Resultado de posicionamiento')
        st.dataframe(result[main_cols + prob_cols], use_container_width=True)

        st.subheader('Perfil multidimensional')
        profile_cols = [c for c in ['sujeto','Nivel_estimado_MCER'] + DIMENSIONS if c in result.columns]
        st.dataframe(result[profile_cols], use_container_width=True)

        if len(result) == 1:
            vals = result.iloc[0][DIMENSIONS]
            st.bar_chart(vals)

        st.download_button(
            'Descargar resultados en Excel',
            data=to_excel_bytes(result),
            file_name=f'TRUNA_ELE_Navigator_{mode_key}_resultados.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    else:
        st.info('Sube un archivo o pega un texto para iniciar la demostración.')
except Exception as exc:
    st.error(f'No fue posible procesar la entrada: {exc}')

