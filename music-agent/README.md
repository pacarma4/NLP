# 🎵 Music Agent

Agente musical que recomienda canciones razonando paso a paso.
Usa NLP (TF-IDF + similitud coseno) sobre un dataset de letras de canciones
y un loop agéntico (ReAct) para decidir qué herramientas usar.

## Cómo funciona

```
Usuario: "Quiero algo melancólico con guitarras"
    │
    ▼
Agente (LLM) razona:
    │
    ├── 1. get_user_profile() → analiza tu historial
    ├── 2. search_songs()     → busca en dataset por TF-IDF
    ├── 3. filter_by()        → filtra por género/artista
    │
    ▼
Recomendación final con justificación
```

## Setup

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configurar tu API key

Edita `agent.py` y cambia:

```python
OPENAI_CONFIG = {
    "api_key": "TU_API_KEY",
    "base_url": "https://api.moonshot.cn/v1",  # Kimi
    "model": "moonshot-v1-8k",
}
```

Si usas otro proveedor, cambia `base_url`:
- **Kimi**: `https://api.moonshot.cn/v1` → modelo `moonshot-v1-8k`
- **DeepSeek**: `https://api.deepseek.com` → modelo `deepseek-chat`
- **Groq**: `https://api.groq.com/openai/v1` → modelo `llama-3.1-70b-versatile`
- **OpenAI**: `https://api.openai.com/v1` → modelo `gpt-4o-mini`

### 3. Preprocesar el dataset

```bash
cp /ruta/a/dataset_fusionado.csv .
python preprocess.py
```

Esto genera la carpeta `data/` con los vectores TF-IDF.

### 4. Lanzar la app

```bash
streamlit run app.py
```

## Estructura

```
music-agent/
├── preprocess.py      # Limpieza + lematización + TF-IDF (ejecutar 1 vez)
├── tools.py           # Herramientas del agente (search, profile, filter)
├── agent.py           # Loop agéntico ReAct + config del LLM
├── app.py             # Interfaz Streamlit
├── requirements.txt
├── data/              # (generado por preprocess.py)
│   ├── songs.pkl
│   ├── tfidf.pkl
│   └── tfidf_matrix.pkl
└── dataset_fusionado.csv
```

## El agente paso a paso

El agente sigue un loop **ReAct** (Reason + Act):

1. **Think** — el LLM razona qué necesita hacer
2. **Act** — decide llamar a una herramienta
3. **Observe** — recibe el resultado
4. **Repeat** — hasta que tiene suficiente info para responder

Todos los pasos son visibles en la interfaz de Streamlit.
