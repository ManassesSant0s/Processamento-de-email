# utils.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
from PyPDF2 import PdfReader

# Baixa stopwords se não existir
try:
    STOPWORDS = set(stopwords.words('portuguese'))
except Exception:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('portuguese'))

lemmatizer = WordNetLemmatizer()
tokenizer = WordPunctTokenizer()  # tokenizer seguro para português

def extract_text_from_pdf(path_or_fileobj):
    reader = PdfReader(path_or_fileobj)
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)

def preprocess_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zà-ú0-9\s]', ' ', text)
    # Tokenização usando WordPunctTokenizer
    tokens = tokenizer.tokenize(text)
    # Remove stopwords e palavras curtas
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    # Lematização
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmas)
