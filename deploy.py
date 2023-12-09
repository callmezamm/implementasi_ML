import streamlit as st
from streamlit_option_menu import option_menu
import joblib

# crawling
import requests
from bs4 import BeautifulSoup
import csv

# normalisasi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import pandas as pd
import numpy as np
import re
import warnings
from nltk.stem import PorterStemmer

# VSM
from sklearn.feature_extraction.text import TfidfVectorizer

# LDA
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import os

# modeling
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib 
from joblib import load


st.header('Implementasi', divider='rainbow')
inputan = st.text_area("Masukkan Berita")
inputan = [inputan]
# ======================== TF-IDF ==========================
vertorizer = load('tf_idf_Vectorizer.pkl')
# ======================== LDA =============================
lda = load('best_lda4.pkl')
# ======================== MODEL ===========================
model = load('best_model4_dc.pkl')

if st.button("Prediksi"):
    ver_inp = vertorizer.transform(inputan)
    lda_inp = lda.transform(ver_inp)
    model_inp = model.predict(lda_inp)
    st.write(model_inp)