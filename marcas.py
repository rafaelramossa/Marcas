from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump
import nltk
import pandas as pd
import pickle

stopwords = nltk.corpus.stopwords.words('portuguese')

# Construindo o modelo SVM com Pipeline
modelo = Pipeline([('vect', TfidfVectorizer()), ('clf', SVC(kernel = 'linear', probability = True))])

# Importando o dataset
df = pd.read_csv("marcas.csv") 

labels = [row for row in df['marca'].str.strip()]
data = [row for row in df['estabelecimento'].str.strip()]

# Objeto para Normalização dos labels
encoder = LabelEncoder()

# Normalização dos labels
target = encoder.fit_transform(labels)

encoder.fit(labels)
le_dict = dict(zip(encoder.transform(encoder.classes_),encoder.classes_))

# salvando as labels
output = open('labels.pkl', 'wb')
pickle.dump(le_dict, output)
output.close()

# Fit do modelo
model = modelo.fit(data, target)

# Salvando o modelo
filename = 'model.sav'
dump(model, filename)



