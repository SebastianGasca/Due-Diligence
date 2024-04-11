#Generales
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re
import unicodedata
import ast
from tqdm import tqdm
from google.colab import files
from collections import defaultdict
import warnings

#Extracción de noticias
import feedparser
from newspaper import Article
from datetime import datetime, timedelta
import urllib.parse

#Sentimientos y Resumen
from transformers import pipeline
import torch
from transformers import BertTokenizerFast, EncoderDecoderModel

#LDA
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')

import spacy
from gensim import corpora
from gensim.models import LdaModel
from pprint import pprint
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.coherencemodel import CoherenceModel

from googletrans import Translator

#barra de carga
from IPython.display import HTML, display
import time

def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))

# ii = 0
def web_scrapping(companies, now, one_day_ago):
    news_data = []
    # Iterar sobre cada empresa en la lista.
    for company in companies:
        print(f"REALIZANDO WEB SCRAPPING PARA -> {company}")
        # Codificar la consulta de búsqueda para manejar espacios y caracteres especiales.
        query = urllib.parse.quote_plus(f"{company} Chile")

        # Construir la URL de búsqueda en Google News para la empresa, asegurándose de codificar la consulta.
        search_url = f"https://news.google.com/rss/search?q={query}&hl=es-419&gl=CL&ceid=CL%3Aes-419"

        # Analizar la fuente RSS.
        news_feed = feedparser.parse(search_url)

        # Iterar sobre cada entrada en el feed.
        for entry in tqdm(news_feed.entries):
            # Obtener la fecha de publicación de la noticia.
            published_date = datetime(*entry.published_parsed[:6])


            # Verificar si la fecha de publicación está dentro del último día.
            if published_date >= one_day_ago and published_date <= now:
                try:
                    # Usar Newspaper3k para extraer el contenido de la noticia.
                    article = Article(entry.link, language='es')
                    article.download()
                    article.parse()

                    # Agregar datos a la lista de noticias.
                    news_data.append({
                        'Empresa': company,
                        'Título': entry.title,
                        'Enlace': entry.link,
                        'Fecha de la noticia': published_date,
                        'Contenido de la noticia': article.text
                    })
                except Exception as e:
                    # Capturar excepciones y mostrar un mensaje de error.
                    print(f"Error al procesar la noticia: {e}")

    # ii = 15
    # out.update(progress(ii, 100))
    print("WEB SCRAPPING COMPLETADO CON EXITO")
    print("\n")

    # Crear el DataFrame.
    noticias = pd.DataFrame(news_data)
    noticias["Texto completo"] = noticias["Título"] + ". " + noticias["Contenido de la noticia"]