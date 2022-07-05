import speech_recognition as sr
from gtts import gTTS
import playsound
import os
import random
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def clean_str(r):
    r = r.lower()
    r = [c for c in r if c in alphabet]
    return ''.join(r)

alphabet = ' 1234567890-йцукенгшщзхъфывапролджэячсмитьбюёqwertyuiopasdfghjklzxcvbnm'

with open('dialogues.txt', encoding='utf-8') as f:
    content = f.read()

blocks = content.split('\n')
dataset = []
for block in blocks:
    replicas = block.split('\\')[:2]
    if len(replicas) == 2:
        pair = [clean_str(replicas[0]), clean_str(replicas[1])]
        if pair[0] and pair[1]:
            dataset.append(pair)

X_text = []
y = []
for question, answer in dataset[:10000]:
    X_text.append(question)
    y += [answer]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_text)
clf = LogisticRegression()
clf.fit(X, y)

def get_generative_replica(text):
    text_vector = vectorizer.transform([text]).toarray()[0]
    question = clf.predict([text_vector])[0]
    return question