import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from tqdm import tqdm
tqdm.pandas() 

# Cargar el CSV
df = pd.read_csv('Tweets.csv', encoding='latin-1')

#========== Análisis Exploratorio de Datos (EDA) ==========#
# Ver estructura básica
print(df.head())
print(df.info())
print(df['sentiment'].value_counts())  # Columna objetivo (sentimiento)

# Visualizar distribución de sentimientos
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='sentiment', order=df['sentiment'].value_counts().index)
plt.title('Distribución de Sentimientos en Tweets')
plt.xlabel('Sentimiento')
plt.ylabel('Cantidad de Tweets')
plt.show()
#===========================================================#

#========== Preprocesamiento de Texto ==========#
# Cargar modelo de spaCy
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

print("\nLimpieza y lematización en progreso...")
df['processed_text'] = df['text'].progress_apply(clean_text)
print(df[['text', 'processed_text']].head(10))
