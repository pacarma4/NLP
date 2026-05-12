"""
preprocess.py — Limpia el dataset, lematiza letras y genera el índice TF-IDF.
Ejecutar UNA VEZ antes de lanzar la app:
    python preprocess.py
"""

import pandas as pd
import pickle
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# ─── Configuración ───────────────────────────────────────────
DATASET_PATH = "outputs/songs_with_years_processed.csv"
OUTPUT_DIR = "data"

# ─── 1. Cargar y limpiar ─────────────────────────────────────
print("📂 Cargando dataset...")
df = pd.read_csv(DATASET_PATH)

# Eliminar columna basura del merge de git
df = df.drop(columns=["<<<<<<< HEAD"], errors="ignore")

# Quedarnos solo con filas que tienen letra Y género
df = df.dropna(subset=["lyrics", "genre_clean"])
df = df.reset_index(drop=True)

print(f"   → {len(df)} canciones con letras y género")
print(f"   → {df['artist'].nunique()} artistas")
print(f"   → {df['genre_clean'].nunique()} géneros")

# ─── 2. Preprocesar letras ───────────────────────────────────
print("\n🔧 Preprocesando letras...")


def clean_lyrics(text):
    """Limpieza básica de letras."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)           # [Chorus], [Verse 1]...
    text = re.sub(r"\(.*?\)", "", text)            # (x2), (repeat)...
    text = re.sub(r"[^a-záéíóúñü\s]", " ", text)  # solo letras
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["lyrics_clean"] = df["lyrics"].apply(clean_lyrics)

# Filtrar letras demasiado cortas
df = df[df["lyrics_clean"].str.len() > 50]
df = df.reset_index(drop=True)
print(f"   → {len(df)} canciones tras limpieza")

# ─── 3. Lematización ─────────────────────────────────────────
# Usa spaCy si está instalado, si no usa las letras limpias directamente.
# Para mejor calidad: pip install spacy && python -m spacy download en_core_web_sm

try:
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    print("\n📝 Lematizando con spaCy (puede tardar unos minutos)...")

    def lemmatize(text):
        doc = nlp(text)
        return " ".join(
            [t.lemma_ for t in doc if not t.is_stop and len(t.lemma_) > 2]
        )

    # Procesar en batches para eficiencia
    batch_size = 1000
    lemmatized = []
    for i in range(0, len(df), batch_size):
        batch = df["lyrics_clean"].iloc[i : i + batch_size].tolist()
        docs = nlp.pipe(batch, batch_size=128)
        for doc in docs:
            lemmatized.append(
                " ".join(
                    [t.lemma_ for t in doc if not t.is_stop and len(t.lemma_) > 2]
                )
            )
        print(f"   ... {min(i + batch_size, len(df))}/{len(df)}")
    df["lyrics_lemmatized"] = lemmatized

except (ImportError, OSError):
    print("\n⚠️  spaCy no disponible — usando letras limpias sin lematizar.")
    print("   Instala para mejor calidad: pip install spacy && python -m spacy download en_core_web_sm")
    df["lyrics_lemmatized"] = df["lyrics_clean"]

# ─── 4. Vectorización TF-IDF ─────────────────────────────────
print("\n🧮 Generando vectores TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.95,
    stop_words="english",
    ngram_range=(1, 2),
)
X = tfidf.fit_transform(df["lyrics_lemmatized"])
print(f"   → Matriz TF-IDF: {X.shape}")

# ─── 5. Guardar ──────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

df.to_pickle(f"{OUTPUT_DIR}/songs.pkl")
pickle.dump(tfidf, open(f"{OUTPUT_DIR}/tfidf.pkl", "wb"))
pickle.dump(X, open(f"{OUTPUT_DIR}/tfidf_matrix.pkl", "wb"))

print(f"\n✅ Todo guardado en {OUTPUT_DIR}/")
print("   → songs.pkl         (metadatos de canciones)")
print("   → tfidf.pkl         (vectorizador entrenado)")
print("   → tfidf_matrix.pkl  (matriz de vectores)")
print("\nAhora puedes lanzar: streamlit run app.py")
