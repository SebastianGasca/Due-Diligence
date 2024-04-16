#Generales
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re
import unicodedata
from pprint import pprint
import ast
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
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
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')
nltk.download('punkt')

import spacy
from gensim import corpora
from gensim.models import LdaModel
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

# FUNCIONES DE LIMPIEZA DE TEXTO Y CREACION DE COLUMNAS
def estandarizar_texto(texto):
    texto = texto.lower()     # Convertir a minúsculas
    texto = re.sub(r'[^\w\s]', '', texto)     # Eliminar signos de puntuación utilizando expresiones regulares
    texto = ''.join( (c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn') )     # Eliminar tildes y diacríticos
    return texto


def contar_palabras(documento, palabras_clave):
    contador = 0
    palabras_encontradas = []
    for palabra_clave in palabras_clave:
        palabra_clave = estandarizar_texto(palabra_clave)
        if palabra_clave in documento.split():
            contador += 1
            palabras_encontradas.append(palabra_clave)
    return contador, palabras_encontradas


def columna_semanas(noticias):
    noticias["semana"] = (noticias["Fecha de la noticia"].dt.isocalendar().week).astype(str)
    noticias["año"] = (noticias["Fecha de la noticia"].dt.year).astype(str)

    def formato_semana(df):
        if len(df["semana"]) == 1:
            return df["año"] + "-" + "0" + df["semana"]
        else:
            return df["año"] + "-" + df["semana"]

    noticias["semana_del_año"] =  noticias.apply(formato_semana, axis=1)
    return noticias

def estandarizar_texto_portokens(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stopwords.words('spanish')]
    return tokens


# FUNCIONES DE PROCESO
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
        for entry in tqdm_notebook(news_feed.entries):
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

    print("WEB SCRAPPING COMPLETADO CON EXITO")
    print("\n")

    # Crear el DataFrame.
    noticias = pd.DataFrame(news_data)
    noticias["Texto completo"] = noticias["Título"] + ". " + noticias["Contenido de la noticia"]
    
    return noticias
    


def palabras_en_noticias(noticias, palabras_clave):
    tqdm.pandas()
    noticias["Texto estandarizado"] = noticias["Texto completo"].apply(estandarizar_texto)
    
    # Aplicar la función contar_palabras a cada elemento de la columna "Texto estandarizado"
    c , p = zip(*noticias["Texto estandarizado"].progress_apply(lambda x: contar_palabras(x, palabras_clave)))
    noticias["N Palabras Encontradas"], noticias["Palabras Encontradas"] = c , p
    
    print("\n CONTADOR DE PALABRAS COMPLETADO CON EXITO")
    return noticias
    

def resumir_noticias(noticias):
    print("RESUMIENDO NOTICIAS CON -bert2bert-")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = 'mrm8488/bert2bert_shared-spanish-finetuned-summarization'
    tokenizer = BertTokenizerFast.from_pretrained(ckpt)
    model = EncoderDecoderModel.from_pretrained(ckpt).to(device)
    
    def generate_summary(text):
        inputs = tokenizer([text],
                           padding="max_length",
                           truncation=True,
                           max_length=512,
                           return_tensors="pt")

        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        output = model.generate(input_ids, attention_mask=attention_mask)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    
    # progress_bar = tqdm(total=noticias.shape[0])
    resumenes_noticias = []
    progress_bar = tqdm(total=noticias.shape[0])
    for index, noticia in enumerate(noticias.iterrows()):
        texto = noticia[1]["Texto completo"]
        resumen = generate_summary(texto)
        resumenes_noticias.append(resumen)
        progress_bar.update(1)
    
    noticias["resumen"] = resumenes_noticias
    
    print("\n RESUMEN DE NOTICIAS COMPLETADO CON EXITO")
    print("\n")
    
    return noticias
    

def evaluando_sentimientos(noticias):
    print("ANALISANDO SENTIMIENTO DE NOTICIAS CON -distilbert-")
    nlp_model = pipeline(model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    sentimientos = []

    progress_bar = tqdm_notebook(total=noticias.shape[0])
    for index, noticia in enumerate(noticias.iterrows()):
        progress_bar.update(1)
        try:
            corpus = noticia[1]['Contenido de la noticia'][:1000]
            sentiment = nlp_model(corpus)
            sentimientos.append(sentiment[0]["label"])
            progress_bar.update(1)
        except:
            sentimientos.append(None)
            progress_bar.update(1)

    noticias["sentimientos"] = sentimientos

    print("ANALISIS DE SENTIMIENTOS COMPLETADO CON EXITO")
    print("\n")
    
    return noticias


def creando_lda(noticias): 
    print("GENERANDO LDA")
    
    nlp = spacy.load("es_core_news_sm")
    documentos = list(noticias["Texto estandarizado"])

    # Preprocesamiento de los documentos
    processed_docs = [estandarizar_texto_portokens(doc) for doc in documentos]
    # Crear un diccionario a partir de los documentos
    diccionario = corpora.Dictionary(processed_docs)
    # Crear el corpus
    corpus = [diccionario.doc2bow(documento) for documento in processed_docs]

    print("ENCONTRANDO NUMERO DE CLUSTER OPTIMOS")
    
    coherence_scores = []
    progress_bar = tqdm(total=len(range(2, 11)))
    for num_topics in range(2, 11):  # Prueba de 2 a 10 tópicos
        progress_bar.update(1)
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=diccionario, passes=10)
        coherence_model = CoherenceModel(model=lda_model, texts=processed_docs,
                                        dictionary=diccionario, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        coherence_scores.append(coherence_score)

    n_topicos = coherence_scores.index(max(coherence_scores)) + 2
    
    print(f"\n CREANDO LDA CON {n_topicos} TOPICOS")
    lda_modelo = LdaModel(corpus, num_topics=n_topicos, id2word=diccionario, passes=15)
    return documentos, corpus, lda_modelo


def topicos_agrupados(lda_modelo):
    # Listas para almacenar los datos
    topicos = []
    palabras = []
    factores = []

    # Expresión regular para encontrar palabras y factores
    regex_palabra = re.compile(r'"\b(\w+)\b"')
    regex_numero = re.compile(r'\d+\.\d+')

    sentencias = lda_modelo.print_topics(num_words = 30); sentencias

    # Extraer palabras y factores de cada sentencia
    print(f"CREANDO DATAFRAME CON TOPCIOS_LDA ")
    for i, sentencia in enumerate(sentencias):
        sentencia = sentencia[1].split("+")
        for s in sentencia:
            s = s.replace(",", "")
            palabra = regex_palabra.findall(s)
            numero = regex_numero.findall(s)
            try:
                palabra = palabra[0]
                numero = numero[0]
            except:
                palabra = None
                numero = None
            topicos.append(f"Topico: {i}")
            palabras.append(palabra)
            factores.append(numero)

    # Crear DataFrame
    df = pd.DataFrame({"Topico_LDA": topicos, "Palabra": palabras, "Factor": factores})

    stopwords_es = spacy.lang.es.stop_words.STOP_WORDS
    df_filtrado = df[(~df['Palabra'].isin(stopwords_es)) & (~df["Palabra"].isna())]

    df_agrupado = (df_filtrado.groupby('Topico_LDA')
                .apply(lambda x: pd.Series({'Palabra': list(x['Palabra']),
                                            'Factor': list(x['Factor'])}))
                .reset_index())
    df_agrupado.rename(columns = {"Palabra": "Palabras_topico_lda"}, inplace = True)
    
    return df_agrupado


# DE DONDE SACO EL CORPUS ? 
def asignando_topicos(noticias, documentos, corpus, df_agrupado, lda_modelo):
    topicos_l = []
    topico_prob = []

    print(f"ASIGNANDO TOPICOS_LDA AL DATAFRAME NOTICIAS")
    for i, documento in enumerate(documentos):
        temas = lda_modelo.get_document_topics(corpus[i])

        maximo_valor = max(temas, key=lambda x: x[1])[1]
        topico_probale = max(temas, key=lambda x: x[1])[0]

        topicos_l.append(f"Topico: {topico_probale}")
        topico_prob.append(maximo_valor)

    noticias["Topico_LDA"] = topicos_l
    noticias["Prob_Topico"] = topico_prob

    noticias_con_lda = noticias.merge(df_agrupado, on = "Topico_LDA")

    print("LDA COMPLETADO CON EXITO")
    print("\n")
    
    return noticias_con_lda


def normalizando_palabras_topicos(practicas, df_agrupado):
    print(f"NORMALIZANDO PALABRAS EN TOPICOS PROPIOS")
    nlp = spacy.load("en_core_web_sm")
    translator = Translator() # TRADUCIMOS AL INGLES
    stemmer = PorterStemmer()
    l = []
    progress_bar = tqdm(total=practicas.shape[0])

    for index, row in practicas.iterrows():
        progress_bar.update(1)
        list_acciones = []
        for palabra in row.acciones:
            palabra_en = translator.translate(palabra, dest="en").text
            palabra_stem = stemmer.stem(palabra_en)
            palabra_lemma = nlp(palabra_en)[0].lemma_
            try:
                palabra_es = translator.translate(palabra_lemma, dest="es").text
            except:
                palabra_es = palabra
                list_acciones.append(palabra_es.lower())

    l.append( (row.practicas , list_acciones) )
    topicos_propios = pd.DataFrame(l, columns = ["practicas", "acciones"])
    
    print("\n")
    print(f"NORMALIZANDO PALABRAS EN TOPICOS LDA")  
    translator = Translator()
    stemmer = PorterStemmer()
    l = []
    progress_bar = tqdm(total=df_agrupado.shape[0])

    for index, row in df_agrupado.iterrows():
        progress_bar.update(1)
        list_acciones = []
        for palabra in row.Palabras_topico_lda:
            try:
                palabra_en = translator.translate(palabra, dest="en").text
            except:
                palabra_en = palabra
                palabra_lemma = nlp(palabra_en)[0].lemma_

            try:
                palabra_es = translator.translate(palabra_lemma, dest="es").text
            except:
                palabra_es = palabra

            list_acciones.append(palabra_es.lower())

    l.append( (row.Topico_LDA , list_acciones) )
    topicos_lda = pd.DataFrame(l, columns = ["Topico_LDA", "Palabras_topico_lda"])
    
    print("\n PALABRAS NORMALIZADAS COMPLETADO CON EXITO") 
    
    return topicos_propios, topicos_lda

def noticias_match_topicos(topicos_lda, topicos_propios, noticias_con_lda):
    print("GENERNADO DATAFRAME FINAL")
    dicc = defaultdict(list)
    for name, row in topicos_lda.iterrows():
      topico = row["Topico_LDA"]
      palabras_topico = row["Palabras_topico_lda"]

      for name_p, row_p in topicos_propios.iterrows():
        pra = row_p["practicas"]
        acc = row_p["acciones"]

        if any(accion in palabras_topico for accion in acc):
          dicc[topico].append(pra)

    l_topic = []
    for k,v in dicc.items():
      l_topic.append( (k,v) )

    topicos_match = pd.DataFrame(l_topic, columns = ["Topico_LDA", "Topico_propio"])

    noticias_que_coinciden = noticias_con_lda[ noticias_con_lda["Topico_LDA"].isin(topicos_match.Topico_LDA) ]
    noticias_que_coinciden = noticias_que_coinciden.merge(topicos_match, on = "Topico_LDA")
    
    print(f"\n DATAFRAME FINAL CONTIENE {noticias_que_coinciden.shape[0]} REGISTROS")
    
    return noticias_que_coinciden