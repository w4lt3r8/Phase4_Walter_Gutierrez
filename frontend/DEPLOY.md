# Guía de despliegue gratuito — Ejercicio 3

## Requisitos previos
1. Haber ejecutado el notebook del **Ejercicio 2** completo en Google Colab.
2. Haber descargado `artefactos_frontend.zip` (generado en la última celda del notebook) y
   descomprimido su contenido dentro de la carpeta `frontend/`, de modo que quede así:

```
frontend/
├── app.py
├── requirements.txt
├── config.yaml                <- del zip
└── kb_faiss_index/             <- del zip
    ├── index.faiss
    └── index.pkl
```

## Opción A — Streamlit Community Cloud (recomendada)
1. Crea un repositorio público en GitHub y sube toda la carpeta `frontend/` (incluyendo
   `config.yaml` y `kb_faiss_index/`).
2. Ingresa a https://share.streamlit.io/ e inicia sesión con tu cuenta de GitHub (gratis).
3. Clic en **"New app"** → selecciona el repositorio y la rama.
4. En **"Main file path"** escribe `app.py`.
5. Clic en **"Deploy"**. La primera vez tardará varios minutos porque descarga el modelo LLM.
6. (Opcional) Si usas el proveedor `hf_api`, agrega tu token en **Settings → Secrets**:
   ```
   HUGGINGFACEHUB_API_TOKEN = "hf_xxx..."
   ```

## Opción B — Hugging Face Spaces
1. Crea una cuenta gratuita en https://huggingface.co/join
2. Ve a **New Space** → elige SDK **Streamlit** → visibilidad pública.
3. Sube (o haz *git push*) los archivos `app.py`, `requirements.txt`, `config.yaml` y
   `kb_faiss_index/` a la raíz del Space.
4. El Space se construye y despliega automáticamente. Revisa los logs si falla el build.
5. Si usas `provider: hf_api`, agrega tu token en **Settings → Repository secrets** como
   `HUGGINGFACEHUB_API_TOKEN`.

## Ejecución local (antes de desplegar, para probar)
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## Notas
- El plan gratuito de Streamlit Community Cloud tiene 1 GB de RAM; si el modelo local
  (Qwen2.5-1.5B) no carga por límite de memoria, cambia `CONFIG["llm"]["provider"]` a
  `"hf_api"` en `config.yaml` para usar la Inference API gratuita de Hugging Face en lugar
  de cargar el modelo localmente.
- Los logs de conversación (`chat_logs.csv`) se generan en el sistema de archivos del
  servicio de despliegue y pueden no persistir entre reinicios gratuitos; para el análisis
  del Ejercicio 3 (protocolo de evaluación), descárgalos periódicamente o ejecuta la app
  localmente durante las pruebas.
