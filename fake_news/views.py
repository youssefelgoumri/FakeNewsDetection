import pickle
import nltk

from nltk.corpus import stopwords

stopwords = stopwords.words('english')
from string import punctuation

from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def supp_punct_ex(ex):
  rmv_punct = ex.translate(str.maketrans ('', '', punctuation))
  return rmv_punct

def tokenization_ex(rmv_punct):
  txt_tokens = nltk.word_tokenize(rmv_punct.lower())
  return txt_tokens

def rmv_stopwords_ex(txt_tokens):
  tokens_clean = []
  for token in txt_tokens:
    if token not in stopwords:
      tokens_clean.append(token)
  return tokens_clean

def lemma_ex(tokens_clean):
  lemma = nltk.WordNetLemmatizer()
  tokens_lemma = []
  for token in tokens_clean:
    token_lemma = lemma.lemmatize(token)
    tokens_lemma.append(token_lemma)
  return tokens_lemma

def clean_ex(tokens):
  sentance = ' '.join(tokens)

  return sentance


def index(request):
    return render(request, "index.html")


x_df = pickle.load(open('prediction/text.pkl', 'rb'))

count_vectorizer = CountVectorizer()
tfidf = TfidfTransformer(norm="l2")
count_vectorizer.fit_transform(x_df)

def result(request):
    model = pickle.load(open('prediction/model.sav', 'rb'))
    model_lr = pickle.load(open('prediction/model_lr.sav', 'rb'))
    model_svm = pickle.load(open('prediction/model_svm.sav', 'rb'))


    if request.method == 'POST':
        title = request.POST['title']
        author = request.POST['author']
        text = request.POST['text']


        #preprocessing
        rmv_punct = supp_punct_ex(text)
        txt_tokens = tokenization_ex(rmv_punct)
        rmv_stp = rmv_stopwords_ex(txt_tokens)
        lemma_tokens = lemma_ex(rmv_stp)
        text_ex = clean_ex(lemma_tokens)


        #tf-idf
        new_freq_term_matrix = count_vectorizer.transform([text_ex])
        tfidf.fit(new_freq_term_matrix)
        new_tf_idf_matrix = tfidf.fit_transform(new_freq_term_matrix)

        # Predict probabilities
        prob_fake = model_svm.predict_proba(new_tf_idf_matrix)[0][1]
        prob_real = model_svm.predict_proba(new_tf_idf_matrix)[0][0]

        prob_fake = prob_fake * 100
        prob_real = prob_real * 100

        if prob_fake > prob_real:
            res = "The news is Fake"
        else:
            res = "The news is Real"

    return render(request, "index.html", context={"title": title,"prediction": res, "prob_fake": prob_fake, "prob_real": prob_real })