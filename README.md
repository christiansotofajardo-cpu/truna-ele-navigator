# TRUNA-ELE Navigator

Sistema demostrativo de posicionamiento lingüístico-discursivo para producciones escritas de Español como Lengua Extranjera (ELE).

## Entrada

- Excel con columnas: sujeto, texto
- Texto pegado directamente

## Salida

- Nivel estimado A1-C1
- Confianza
- Perfil multidimensional
- Excel descargable con resultados

## Despliegue

Build Command:
pip install -r requirements.txt

Start Command:
python -m streamlit run app.py --server.port $PORT --server.address 0.0.0.0
