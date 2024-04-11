def load_functions(): 
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

    ii = 0