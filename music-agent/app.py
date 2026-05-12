"""
app.py — Interfaz Streamlit con pasos del agente visibles.
Ejecutar con: streamlit run app.py
"""

import streamlit as st
import json
import time

st.set_page_config(page_title="Music Agent", page_icon="🎵", layout="centered")

# ─── Estilos ──────────────────────────────────────────────────
st.markdown("""
<style>
    .step-box {
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        border-radius: 8px;
        font-size: 0.9rem;
        border-left: 3px solid;
    }
    .step-thinking {
        background: #f0f0ff;
        border-color: #7c6cf0;
        color: #4a3d8f;
    }
    .step-tool {
        background: #e8f5e9;
        border-color: #4caf50;
        color: #2e7d32;
    }
    .step-result {
        background: #fff3e0;
        border-color: #ff9800;
        color: #e65100;
    }
    .step-done {
        background: #e3f2fd;
        border-color: #2196f3;
        color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────
st.title("🎵 Music Agent")
st.caption("Agente musical con razonamiento visible — busca en tu dataset local")

# ─── Sidebar: historial de escucha ────────────────────────────
with st.sidebar:
    st.header("🎧 Tu historial")
    st.caption("Canciones que has escuchado recientemente")

    # Estado para manejar canciones
    if "historial" not in st.session_state:
        st.session_state.historial = [
            "Lua - Bright Eyes",
            "Motion Picture Soundtrack - Radiohead",
            "Between the Bars - Elliott Smith",
        ]

    # Mostrar canciones actuales
    for i, song in enumerate(st.session_state.historial):
        col1, col2 = st.columns([5, 1])
        col1.text(song)
        if col2.button("✕", key=f"del_{i}"):
            st.session_state.historial.pop(i)
            st.rerun()

    # Añadir canción
    new_song = st.text_input("Añadir canción", placeholder="Título - Artista")
    if st.button("➕ Añadir") and new_song:
        st.session_state.historial.insert(0, new_song)
        st.rerun()

    st.divider()
    st.caption("El agente buscará estas canciones en el dataset para entender tu gusto.")

# ─── Input principal ──────────────────────────────────────────
query = st.text_area(
    "¿Qué estilo buscas?",
    placeholder="Ej: Algo melancólico con guitarras acústicas, para una noche tranquila...",
    height=100,
)

# ─── Ejecutar agente ──────────────────────────────────────────
if st.button("▶ Ejecutar agente", type="primary", use_container_width=True):
    if not query:
        st.warning("Escribe qué estilo de canción buscas")
        st.stop()

    # Construir el input completo para el agente
    historial_str = ", ".join(st.session_state.historial)
    full_input = (
        f"Historial de escucha del usuario: {historial_str}\n\n"
        f"Estilo que busca: {query}"
    )

    # Contenedor para los pasos
    steps_container = st.container()
    response_container = st.container()

    # Importar el agente aquí (para que no falle si no hay data/)
    try:
        from agent import run_agent
    except Exception as e:
        st.error(f"Error cargando el agente: {e}")
        st.info("¿Has ejecutado `python preprocess.py` primero?")
        st.stop()

    # Callback para mostrar pasos en tiempo real
    step_count = {"n": 0}

    def show_step(step_type, data):
        step_count["n"] += 1
        with steps_container:
            if step_type == "thinking":
                st.markdown(
                    f'<div class="step-box step-thinking">'
                    f'🧠 <b>Paso {data["iteration"]}</b> — Razonando...'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            elif step_type == "tool_call":
                # Nombres descriptivos para cada herramienta
                tool_labels = {
                    "search_songs": "🔍 Buscando canciones en el dataset",
                    "get_user_profile": "👤 Analizando tu perfil de escucha",
                    "filter_by": "✂️ Filtrando resultados",
                }
                label = tool_labels.get(data["name"], f"🔧 {data['name']}")
                args_str = ", ".join(f"{k}={v}" for k, v in data["args"].items())

                st.markdown(
                    f'<div class="step-box step-tool">'
                    f'<b>{label}</b><br>'
                    f'<code>{data["name"]}({args_str})</code>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            elif step_type == "tool_result":
                # Parsear resultado para mostrarlo bonito
                try:
                    result = json.loads(data["result"])
                    if isinstance(result, list):
                        summary = f"{len(result)} resultados encontrados"
                    elif isinstance(result, dict) and "genero_dominante" in result:
                        summary = (
                            f"Género dominante: {result['genero_dominante']} · "
                            f"Keywords: {', '.join(result.get('keywords_gusto', [])[:5])}"
                        )
                    else:
                        summary = data["result"][:200]
                except (json.JSONDecodeError, TypeError):
                    summary = data["result"][:200]

                st.markdown(
                    f'<div class="step-box step-result">'
                    f'📊 <b>Resultado de {data["name"]}</b>: {summary}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            elif step_type == "final":
                st.markdown(
                    f'<div class="step-box step-done">'
                    f'✅ <b>Agente terminó</b> en {step_count["n"]} pasos'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Ejecutar
    with st.spinner("El agente está trabajando..."):
        try:
            result = run_agent(full_input, on_step=show_step)
        except Exception as e:
            st.error(f"Error del agente: {e}")
            st.stop()

    # Mostrar respuesta final
    with response_container:
        st.divider()
        st.subheader("🎵 Recomendación")
        st.markdown(result)
