import pandas as pd

# 1. Leer el archivo indicando que está dentro de la subcarpeta
dataset_fusionado = pd.read_csv('dataset_fusionado.csv')

# 2. Filtrar las columnas
dataset_reducido = dataset_fusionado[['artist', 'song']]

# 3. Guardar el resultado dentro de esa misma subcarpeta
dataset_reducido.to_csv('dataset_fusionado_reducido.csv', index=False)

print("¡Archivo guardado con éxito!")