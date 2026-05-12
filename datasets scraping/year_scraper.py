import pandas as pd
import musicbrainzngs
import time
import os

# --- 1. CONFIGURACIÓN ---
# Usamos tu archivo original (el que tenía repetidas)
INPUT_FILE = 'dataset_fusionado.csv' 
PROGRESS_FILE = 'canciones_progreso.csv'
FINAL_FILE = 'dataset_final_con_años.csv'

# MusicBrainz exige que pongamos un nombre de app y un correo
musicbrainzngs.set_useragent("Buscador_Proyectos_Datos", "1.0", "tu_correo@gmail.com")

# --- 2. PREPROCESADO PARA LA API ---
print("Cargando y limpiando dataset...")
df_raw = pd.read_csv(INPUT_FILE, on_bad_lines='skip')

# Usamos las columnas de artist y song
df_api = df_raw[['artist', 'song']].copy()

# Limpieza de los datos (quitar los '/' de los artistas como en "/moonspell/")
df_api['artist'] = df_api['artist'].astype(str).str.strip('/').str.replace('-', ' ')
df_api = df_api.dropna()

# *** CLAVE AQUÍ ***: Sacamos pares únicos para no buscar la misma canción 2 veces
unique_pairs = df_api.drop_duplicates().reset_index(drop=True)

# --- 3. REANUDAR PROGRESO (Por si lo paras o se corta la luz) ---
if os.path.exists(PROGRESS_FILE):
    processed_df = pd.read_csv(PROGRESS_FILE)
    merged = unique_pairs.merge(processed_df[['artist', 'song']], on=['artist', 'song'],
                               how='left', indicator=True)
    to_process = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    results = processed_df.to_dict('records')
    print(f"Resumiendo: Faltan {len(to_process)} de {len(unique_pairs)} canciones únicas.")
else:
    to_process = unique_pairs
    results = []
    print(f"Iniciando desde cero: {len(to_process)} canciones únicas detectadas.")

# --- 4. FUNCIÓN Y BUCLE DE API ---
def fetch_year(artist, song):
    try:
        res = musicbrainzngs.search_recordings(artist=artist, recording=song, limit=1)
        if res['recording-list']:
            rec = res['recording-list'][0]
            if 'release-list' in rec:
                for rel in rec['release-list']:
                    if 'date' in rel: 
                        return rel['date'][:4] # Extraemos solo el año
    except Exception:
        return None
    return "Unknown"

print("\nIniciando búsqueda (Esto tomará un par de horas, puedes dejarlo minimizado)...")
try:
    for index, row in to_process.iterrows():
        year = fetch_year(row['artist'], row['song'])
        results.append({'artist': row['artist'], 'song': row['song'], 'year': year})

        print(f"\r[{len(results)}/{len(unique_pairs)}] {row['song'][:20]}... -> {year}", end="")

        # Guardado constante cada vez que encuentra una canción
        pd.DataFrame(results).to_csv(PROGRESS_FILE, index=False)
        
        # EL TIEMPO DE ESPERA OBLIGATORIO DE MUSICBRAINZ
        time.sleep(1.1)
except KeyboardInterrupt:
    print("\nProceso pausado por el usuario. Puedes volver a ejecutarlo más tarde para continuar.")

# --- 5. FUSIÓN FINAL ---
print("\n\nGuardando archivo final...")
# Unimos los años encontrados a tu dataset inicial
final_df = df_api.merge(pd.DataFrame(results), on=['artist', 'song'], how='left')
final_df.to_csv(FINAL_FILE, index=False)
print(f"¡Hecho! Dataset final guardado como '{FINAL_FILE}'")