import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import calendar
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly import tools
from plotly.offline import plot, iplot
from textblob import TextBlob
import re
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
from statsmodels.tsa.stattools import adfuller

st.title("Analise de Avaliaçãoes")

image = Image.open('echodot.jpg')
st.image(image, caption='', width=350)

st.write("Echo Dot - 4 Geração - Amazon")

dataset = pd.read_csv('NoStem_TotalStopwordRemoval_dataset.csv')
dataset = dataset.dropna()
dataset = dataset.reset_index(drop=True)
    
st.header('Plot dos Dados')
summarised_results = dataset["sent_rating"].value_counts()
qtd_star = ('Positiva', 'Negativa')
plt.bar(qtd_star, summarised_results.values)
plt.title("Tipo de Avaliação")
plt.ylabel("Quantidade de Avaliação")
plt.show()
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)


##################################################################################################################################

#WordClouds
def create_Word_Corpus(dataset):
    comment_words = ''
    for val in dataset["processed_text_NoStopwords"]:
        val = str(val)
        tokens = val.split()
        
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
            comment_words += " ".join(tokens)+" "
            
    return comment_words


def plot_Cloud(wordCloud):
    fig, ax = plt.subplots(figsize=(20,10), facecolor='k')
    plt.imshow(wordCloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    st.pyplot(fig)

    
positivo = dataset.loc[dataset['sent_rating'] == 'positivo']
negativo = dataset.loc[dataset['sent_rating'] == 'negativo']

pos_wordcloud = WordCloud(width=900, height=500, stopwords = stopwords.words('portuguese')).generate(create_Word_Corpus(positivo))
neg_wordcloud = WordCloud(width=900, height=500,stopwords = stopwords.words('portuguese')).generate(create_Word_Corpus(negativo))

st.header('Word Cloud das Avaliações Positivas')
plot_Cloud(pos_wordcloud)
st.header('Word Cloud das Avaliações Negativas')
plot_Cloud(neg_wordcloud)

##################################################################################################################################
#Plot por Sentimento do Comentário, mediante Estrelas(Negativo/Positivo)

#dataset['processed_text_NoStopwords'] = dataset['processed_text_NoStopwords'].astype(str)

#dataset['Tamanho_Texto'] = dataset['processed_text_NoStopwords'].apply(len)
#st.header('Sentimento do Comentário')
#ax = sns.swarmplot(x = 'sent_rating', y = 'Tamanho_Texto', data = dataset, alpha = 0.7, palette = 'coolwarm');
#st.pyplot(fig)

##################################################################################################################################
#Verificação do sentimento por Ano, Mês e Dia


dataset['Count'] = dataset['sent_rating']
dataset['data']= pd.to_datetime(dataset['data'])
dataset['Dia']=dataset['data'].dt.day
dataset['Mês']=dataset['data'].dt.month
dataset['Ano']=dataset['data'].dt.year

Sentiment_Year = dataset.groupby(['Ano','sent_rating'])['Count'].count().reset_index()
Positive_Year = Sentiment_Year[Sentiment_Year.sent_rating == 'positivo']
Negative_Year = Sentiment_Year[Sentiment_Year.sent_rating == 'negativo']

st.header('Avaliações positivas ao longo dos anos com base em sentimentos')
Positive_Year.plot(x="Ano",y="Count",kind="bar",title="")
plt.show()
st.pyplot()

st.header('Avaliações negativas ao longo dos anos com base em sentimentos')
Negative_Year.plot(x="Ano",y="Count",kind="bar",title="")
plt.show()
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

#Mês
Sentiment_Month = dataset.groupby(['Mês','sent_rating'])['Count'].count().reset_index()

Positive_Month = Sentiment_Month[Sentiment_Month.sent_rating == 'positivo']
Negative_Month = Sentiment_Month[Sentiment_Month.sent_rating == 'negativo']

st.header('Avaliações positivas ao longo dos meses com base em sentimentos')
Positive_Month.plot(x="Mês",y="Count",kind="line",title="")
plt.show()
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

st.header('Avaliações negativas ao longo dos meses com base em sentimentos')
Negative_Month.plot(x="Mês",y="Count",kind="line",title="")
plt.show()
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)


Meses = dataset.groupby(['Mês'])['Count'].count().reset_index()
Meses['Mês'] = Meses['Mês'].apply(lambda x: calendar.month_name[x])

st.header('Número de Avaliações Por Mês')
Meses.plot(x="Mês",y="Count",kind="bar",title="Número de Avaliações Por Mês")
plt.show()
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

#Dia
dia = dataset.groupby('Dia')['processed_text_NoStopwords'].count().reset_index()
dia['Dia']=dia['Dia'].astype('int64')
dia.sort_values(by=['Dia'])

st.header('Dia vs Contagem de avaliações')
figu, ax = plt.subplots(figsize=(11, 9))
ax = sns.barplot(x = "Dia", y = "processed_text_NoStopwords", data = dia);
st.pyplot(figu)

##################################################################################################################################
#Unigrama - Bigrama - Trigrama

#Filtrando as avaliações
review_pos = dataset[dataset['sent_rating'] == 'positivo'].dropna()
review_neg = dataset[dataset['sent_rating'] == 'negativo'].dropna()


## função personalizada para geração de Unigrama ##
STOPWORDS = stopwords.words('portuguese')
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

## função personalizada para gráfico de barras horizontais ##
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["Tamanho_Texto"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace


## Gráfico de barras de comentários positivos ##
freq_dict = defaultdict(int)
for sent in review_pos["processed_text_NoStopwords"]:
    for word in generate_ngrams(sent,2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "Tamanho_Texto"]
trace0 = horizontal_bar_chart(fd_sorted.head(25), 'green')

## Gráfico de barras de comentários negativos ##
freq_dict = defaultdict(int)
for sent in review_neg["processed_text_NoStopwords"]:
    for word in generate_ngrams(sent,2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "Tamanho_Texto"]
trace1 = horizontal_bar_chart(fd_sorted.head(25), 'brown')

# Criando dois subplots
fig = tools.make_subplots(rows=2, cols=1, vertical_spacing=0.04,horizontal_spacing=0.25,
                          subplot_titles=["Bigrama plot das avaliações positivas", "Bigrama plots das avaliações negativas"
                                          ])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Bigrama Plots")
st.plotly_chart(fig, filename='word-plots')

#Trigrama
freq_dict = defaultdict(int)
for sent in review_pos["processed_text_NoStopwords"]:
    for word in generate_ngrams(sent,3):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "Tamanho_Texto"]
trace0 = horizontal_bar_chart(fd_sorted.head(25), 'green')


freq_dict = defaultdict(int)
for sent in review_neg["processed_text_NoStopwords"]:
    for word in generate_ngrams(sent,3):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "Tamanho_Texto"]
trace1 = horizontal_bar_chart(fd_sorted.head(25), 'brown')


fig = tools.make_subplots(rows=2, cols=1, vertical_spacing=0.04,horizontal_spacing=0.25,
                          subplot_titles=["Trigrama plot das avaliações positivas", "Trigrama plots das avaliações negativas"
                                          ])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)

fig['layout'].update(height=1000, width=800, paper_bgcolor='rgb(233,233,233)', title="Trigrama Plots")


##################################################################################################################################
#Cluster de reviews postivas
positive_reviews_df = dataset[dataset['sent_rating']=='positivo']
positive_reviews_df = positive_reviews_df.reset_index(drop=True)


#constroi uma nova lista para armazenar o texto
clean_desc = []
for w in range(len(positive_reviews_df.processed_text_NoStopwords)):
    desc = positive_reviews_df['processed_text_NoStopwords'][w].lower()
    split_text = desc.split()

    split_text = " ".join(split_text)
    clean_desc.append(split_text)

#TF-IDF vectorizer
tfv = TfidfVectorizer(ngram_range = (1,1))

#Transforma
vec_text = tfv.fit_transform(clean_desc)

words = tfv.get_feature_names()


#Configuração kmeans clustering
kmeans = KMeans(n_clusters = 3, n_init = 1, max_iter=100)


#Ajuste dos dados
kmeans.fit(vec_text)

common_words = kmeans.cluster_centers_.argsort()#[:,-1:-10:-1]
common_words[:,-1:-11:-1]

#Este loop transforma os números de volta em palavras
common_words = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
for num, centroid in enumerate(common_words):
    str(num) + ' : ' + ', '.join(words[word] for word in centroid)

ratings = positive_reviews_df['sent_rating'].unique()


#adicione o rótulo cluster ao dataset
positive_reviews_df['cluster'] = kmeans.labels_

clusters = positive_reviews_df.groupby(['cluster', 'sent_rating']).size()
fig, ax1 = plt.subplots(figsize = (26, 15))
ax1 = sns.heatmap(clusters.unstack(level = 'sent_rating'), ax = ax1, cmap = 'Reds')
ax1.set_xlabel('Avaliação').set_size(18)
ax1.set_ylabel('Cluster').set_size(18)
st.pyplot(fig)


fig, ax = plt.subplots(figsize = (14, 4))
ax = sns.countplot(x='cluster', data=positive_reviews_df).set_title("Contagens de classificação - Reviews Positivos")
st.pyplot(fig)

###############################################################################################################################
#Cluster de reviews negativas
negative_reviews_df = dataset[dataset['sent_rating']=='negativo']
negative_reviews_df = negative_reviews_df.reset_index(drop=True)


#constroi uma nova lista para armazenar o texto
clean_desc = []
for w in range(len(negative_reviews_df.processed_text_NoStopwords)):
    desc = negative_reviews_df['processed_text_NoStopwords'][w].lower()
    split_text = desc.split()

    split_text = " ".join(split_text)
    clean_desc.append(split_text)

#TF-IDF vectorizer
tfv = TfidfVectorizer(ngram_range = (1,1))

#Transforma
vec_text = tfv.fit_transform(clean_desc)

words = tfv.get_feature_names()


#Configuração kmeans clustering
kmeans = KMeans(n_clusters = 3, n_init = 1, max_iter=100)


#Ajuste dos dados
kmeans.fit(vec_text)

common_words = kmeans.cluster_centers_.argsort()#[:,-1:-10:-1]
common_words[:,-1:-11:-1]

#Este loop transforma os números de volta em palavras
common_words = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
for num, centroid in enumerate(common_words):
   str(num) + ' : ' + ', '.join(words[word] for word in centroid)

ratings = negative_reviews_df['sent_rating'].unique()


#adicione o rótulo cluster ao dataset
negative_reviews_df['cluster'] = kmeans.labels_

clusters = negative_reviews_df.groupby(['cluster', 'sent_rating']).size()
fig, ax1 = plt.subplots(figsize = (26, 15))
ax1 = sns.heatmap(clusters.unstack(level = 'sent_rating'), ax = ax1, cmap = 'Reds')
ax1.set_xlabel('Avaliação').set_size(18)
ax1.set_ylabel('Cluster').set_size(18)
st.pyplot(fig)


fig, ax = plt.subplots(figsize = (14, 4))
ax = sns.countplot(x='cluster', data=negative_reviews_df).set_title("Contagens de classificação - Reviews Negativos")
st.pyplot(fig)







