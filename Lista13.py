import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
import nltk
import re
import string
nltk.download('stopwords')
from nltk.corpus import stopwords

# 1. Carregar dados
data = pd.read_csv('train.csv')  # Arquivo obtido do Kaggle
print(data.head())

# 2. Pré-processamento de texto
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

data['comment_text'] = data['comment_text'].astype(str).apply(clean_text)

# 3. Vetorização
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(data['comment_text'])

# 4. Multirrótulo
y = data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Modelo
classifier = BinaryRelevance(RandomForestClassifier(n_estimators=100, random_state=42))
classifier.fit(X_train, y_train)

# 6. Avaliação
y_pred = classifier.predict(X_test)

print("Hamming Loss:", hamming_loss(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=y.columns))

