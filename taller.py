import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from tqdm import tqdm
tqdm.pandas()

#========== Cargar el CSV original o el preprocesado ==========#
if os.path.exists('Tweets_processed.csv'):
    print("Cargando datos preprocesados desde Tweets_processed.csv ...")
    df = pd.read_csv('Tweets_processed.csv', encoding='latin-1')
else:
    print("Cargando Tweets.csv (archivo original)...")
    df = pd.read_csv('Tweets.csv', encoding='latin-1')

#========== Análisis Exploratorio de Datos (EDA) ==========#
print("\n=== ANÁLISIS EXPLORATORIO DE DATOS ===")
print(df.head())
print(df.info())
print("\nConteo de sentimientos:")
print(df['sentiment'].value_counts())  # Columna objetivo (sentimiento)

# Visualizar distribución de sentimientos
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='sentiment', order=df['sentiment'].value_counts().index)
plt.title('Distribución de Sentimientos en Tweets')
plt.xlabel('Sentimiento')
plt.ylabel('Cantidad de Tweets')
plt.show()
#===========================================================#

#========== Preprocesamiento de Texto (solo si no existe el CSV procesado) ==========#
if 'processed_text' not in df.columns:
    print("\n🧹 Iniciando limpieza y lematización con spaCy...")
    nlp = spacy.load("en_core_web_sm")

    def clean_text(text):
        if not isinstance(text, str):
            return ""
        doc = nlp(text.lower())  # convierte a minúsculas
        tokens = [
            token.lemma_  # lematiza
            for token in doc
            if token.is_alpha and not token.is_stop  # elimina números, signos y stopwords
        ]
        return ','.join(tokens)  # tokens separados por comas

    start_time = time.time()  # medir tiempo de ejecución
    df['processed_text'] = df['text'].progress_apply(clean_text)
    total_time = time.time() - start_time

    # Guardar el resultado procesado
    df.to_csv('Tweets_processed.csv', index=False, encoding='latin-1')
    print(f"\nPreprocesamiento completado y guardado en 'Tweets_processed.csv' ({len(df)} tweets procesados).")
    print(f"⏱️ Tiempo total: {total_time:.2f} segundos\n")

else:
    print("\nLos datos ya están preprocesados. Saltando limpieza...")

#========== Mostrar vista previa final ==========#
print("\n=== VISTA PREVIA DEL RESULTADO FINAL ===")
print(df[['text', 'processed_text']])
