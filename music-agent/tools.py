"""
tools.py — Herramientas que el agente puede invocar.
Cada función trabaja sobre el dataset local con NLP (TF-IDF + similitud coseno).
"""

import pandas as pd
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity


# ─── Cargar datos preprocesados ──────────────────────────────
def load_data():
    df = pd.read_pickle("data/songs.pkl")
    tfidf = pickle.load(open("data/tfidf.pkl", "rb"))
    X = pickle.load(open("data/tfidf_matrix.pkl", "rb"))
    return df, tfidf, X


df, tfidf, X = load_data()


# ─── TOOL 1: Buscar canciones por similitud semántica ────────
def search_songs(query: str, top_k: int = 15) -> str:
    """
    Busca canciones en el dataset cuyas letras sean similares
    al texto de la query, usando similitud coseno sobre TF-IDF.
    """
    query_vec = tfidf.transform([query.lower()])
    scores = cosine_similarity(query_vec, X).flatten()

    top_idx = scores.argsort()[-top_k:][::-1]
    results = []
    for idx in top_idx:
        if scores[idx] > 0.01:  # umbral mínimo de relevancia
            row = df.iloc[idx]
            results.append({
                "idx": int(idx),
                "song": row["song"],
                "artist": row["artist"],
                "genre": row["genre_clean"],
                "score": round(float(scores[idx]), 4),
            })

    return json.dumps(results, ensure_ascii=False)


# ─── TOOL 2: Perfil de gusto del usuario ─────────────────────
def get_user_profile(song_list: str) -> str:
    """
    Analiza una lista de canciones del historial del usuario
    y devuelve un perfil con géneros dominantes, artistas y mood.
    """
    # Buscar cada canción en el dataset
    found = []
    for entry in song_list.split(","):
        entry = entry.strip().lower()
        if not entry:
            continue
        # Buscar por título (fuzzy match simple)
        mask = df["song"].str.lower().str.contains(entry.split("-")[0].strip(), na=False)
        matches = df[mask]
        if len(matches) > 0:
            found.append(matches.iloc[0])

    if not found:
        return json.dumps({
            "error": "No se encontraron canciones del historial en el dataset",
            "tip": "Prueba con otros títulos de canciones"
        }, ensure_ascii=False)

    found_df = pd.DataFrame(found)

    # Análisis del perfil
    profile = {
        "canciones_encontradas": len(found),
        "generos": found_df["genre_clean"].value_counts().to_dict(),
        "artistas": found_df["artist"].unique().tolist(),
        "genero_dominante": found_df["genre_clean"].mode().iloc[0] if len(found_df) > 0 else "desconocido",
    }

    # Generar vector promedio del gusto del usuario
    found_indices = found_df.index.tolist()
    if found_indices:
        avg_vector = X[found_indices].mean(axis=0)
        # Buscar top features (palabras más representativas del gusto)
        feature_names = tfidf.get_feature_names_out()
        avg_arr = avg_vector.A1 if hasattr(avg_vector, "A1") else avg_vector.flatten()
        top_features_idx = avg_arr.argsort()[-10:][::-1]
        profile["keywords_gusto"] = [feature_names[i] for i in top_features_idx]

    return json.dumps(profile, ensure_ascii=False)


# ─── TOOL 3: Filtrar canciones por criterios ─────────────────
def filter_by(song_indices: str, genre: str = "", artist: str = "") -> str:
    """
    Filtra una lista de canciones por género y/o artista.
    song_indices: lista JSON de índices del dataset.
    """
    try:
        indices = json.loads(song_indices)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"error": "Formato de índices inválido"})

    filtered = df.iloc[indices].copy()

    if genre:
        genre_lower = genre.lower()
        mask = (
            filtered["genre_clean"].str.lower().str.contains(genre_lower, na=False)
            | filtered["genres"].str.lower().str.contains(genre_lower, na=False)
        )
        filtered = filtered[mask]

    if artist:
        filtered = filtered[
            filtered["artist"].str.lower().str.contains(artist.lower(), na=False)
        ]

    results = []
    for _, row in filtered.iterrows():
        results.append({
            "song": row["song"],
            "artist": row["artist"],
            "genre": row["genre_1"],
            "genres_all": row["genres"],
        })

    return json.dumps(results, ensure_ascii=False)


# ─── Definición de herramientas para la API (formato genérico) ─
# Adapta los nombres de campo según tu modelo (OpenAI, Kimi, Gemini...)
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_songs",
            "description": "Busca canciones en el dataset local cuyas letras sean similares a una query de texto. Usa similitud coseno sobre vectores TF-IDF. Útil para encontrar canciones con temáticas o vocabulario similar al estilo que pide el usuario.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Texto descriptivo del estilo/mood buscado, ej: 'melancholic acoustic guitar love lost'"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Número de resultados a devolver (default 15)",
                        "default": 15
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_profile",
            "description": "Analiza el historial de escucha del usuario buscando las canciones en el dataset. Devuelve géneros dominantes, artistas y keywords representativas del gusto del usuario.",
            "parameters": {
                "type": "object",
                "properties": {
                    "song_list": {
                        "type": "string",
                        "description": "Lista de canciones separadas por coma, ej: 'Lua - Bright Eyes, Creep - Radiohead'"
                    }
                },
                "required": ["song_list"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by",
            "description": "Filtra una lista de canciones candidatas por género y/o artista. Recibe los índices de canciones (del resultado de search_songs) y aplica filtros.",
            "parameters": {
                "type": "object",
                "properties": {
                    "song_indices": {
                        "type": "string",
                        "description": "Lista JSON de índices del dataset, ej: '[12, 45, 89, 234]'"
                    },
                    "genre": {
                        "type": "string",
                        "description": "Género por el que filtrar, ej: 'folk', 'metal'"
                    },
                    "artist": {
                        "type": "string",
                        "description": "Artista por el que filtrar"
                    }
                },
                "required": ["song_indices"]
            }
        }
    }
]

# Mapa nombre → función para ejecución
TOOLS_MAP = {
    "search_songs": search_songs,
    "get_user_profile": get_user_profile,
    "filter_by": filter_by,
}
