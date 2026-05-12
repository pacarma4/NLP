import pandas as pd
import glob
import os


# CONFIGURACIÓN - Edita estas listas con los archivos

# Archivos con estructura: ALink, SName, Lyric, genres, genre_1, genre_2, genre_3
archivos_tipo_genres = [
    "genres_dataset_90057_100000.csv",
    "genres_dataset_90000_100000.csv",
    "genres_dataset_90055_100000.csv",
    "genres_dataset_90056_100000.csv",
    "genres_dataset_180000_190000.csv",
    "genres_dataset_130000_170000.csv", 
]

# Archivos con estructura: artist, song, lyrics, genres, genre_1, genre_2, genre_3
archivos_tipo_music = [
    "music_dataset_40000_50000.csv",
    "music_dataset.csv",
    "music_dataset_20000_30000.csv",
    "music_dataset_30000_40000.csv",
    "music_dataset_50000_70000.csv",
    "music_dataset_jose.csv",
    "music_dataset_jose_2.csv",
    "music_dataset_jose_3.csv",

]

# Carpeta donde están los CSVs (por defecto, la misma carpeta que este script)
CARPETA = os.path.dirname(os.path.abspath(__file__))

# Nombre del archivo de salida
SALIDA = "dataset_fusionado.csv"

# FUSIÓN

dfs = []

# Leer archivos tipo "genres" y renombrar columnas
for archivo in archivos_tipo_genres:
    ruta = os.path.join(CARPETA, archivo)
    df = pd.read_csv(ruta, on_bad_lines='skip')
    df = df.rename(columns={
        "ALink": "artist",
        "SName": "song",
        "Lyric": "lyrics"
    })
    dfs.append(df)
    print(f"✅ Leído (tipo genres): {archivo} — {len(df)} filas")

# Leer archivos tipo "music" (columnas ya correctas)
for archivo in archivos_tipo_music:
    ruta = os.path.join(CARPETA, archivo)
    df = pd.read_csv(ruta, on_bad_lines='skip')
    dfs.append(df)
    print(f"✅ Leído (tipo music):  {archivo} — {len(df)} filas")

# Concatenar todo
merged = pd.concat(dfs, ignore_index=True)

# Guardar
ruta_salida = os.path.join(CARPETA, SALIDA)
merged.to_csv(ruta_salida, index=False)

print(f"\n🎉 Dataset fusionado guardado en: {ruta_salida}")
print(f"   Total filas: {len(merged)}")
print(f"   Columnas:    {list(merged.columns)}")
