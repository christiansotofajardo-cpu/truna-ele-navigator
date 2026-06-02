# TRUNA-ELE Navigator

Prototipo demostrativo de posicionamiento lingüístico-discursivo para Español como Lengua Extranjera (ELE).

## Descripción

TRUNA-ELE Navigator es una herramienta orientada al análisis automatizado de producciones escritas de aprendientes de Español como Lengua Extranjera (ELE).

El sistema permite procesar textos narrativos o argumentativos y generar una estimación de posicionamiento dentro del continuo de niveles A1–C1, junto con un perfil multidimensional de características lingüísticas y discursivas.

Esta versión corresponde a un demostrador funcional orientado a investigación, transferencia tecnológica y desarrollo futuro de sistemas de evaluación automatizada basados en TRUNAJOD.

---

## Funcionalidades actuales

- Análisis de textos individuales.
- Procesamiento por lotes mediante archivos Excel.
- Estimación de nivel A1–C1.
- Cálculo de confianza de clasificación.
- Perfil multidimensional de desempeño.
- Exportación de resultados en Excel.

---

## Flujo general

Texto

↓

Análisis lingüístico

↓

Perfil multidimensional

↓

Posicionamiento estimado

↓

Reporte de resultados

---

## Próximas versiones

### TRUNA-ELE Navigator v2

- Integración de índices predictivos derivados de estudios TRUNA-ELE.
- Motores diferenciados para tareas narrativas y argumentativas.
- Modelos específicos por tipo de producción.

### TRUNA-ELE Navigator v3

- Retroalimentación automatizada.
- Perfil de fortalezas y oportunidades de mejora.
- Visualizaciones avanzadas.

### TRUNA-ELE Suite

- Nivelador.
- Analizador multidimensional.
- Retroalimentación pedagógica.
- Seguimiento longitudinal.
- Herramientas para investigación en ELE.

---

## Tecnología

- Python
- Streamlit
- Pandas
- NumPy
- OpenPyXL

---

## Despliegue en Render

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
python -m streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

---

## Estado

Versión actual:

**TRUNA-ELE Navigator v1.6 (Demo funcional)**

Proyecto en desarrollo dentro de la línea de investigación TRUNA-ELE y MetaSistema.
