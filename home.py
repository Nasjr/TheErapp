import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from matplotlib import style
from PIL import Image
from arabic_reshaper import arabic_reshaper
from bidi.algorithm import get_display
import nltk
import string
# import streamlit_wordcloud as wordcloud
from wordcloud import WordCloud
nltk.download('stopwords')
from collections import Counter
from streamlit_jupyter import StreamlitPatcher, tqdm
from streamlit_option_menu import option_menu
sns.set(font='DejaVu Sans')
import streamlit as st
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

# marawn_pablo_cleaned=pd.read_csv(r'E:\Projects\Egyptian Rap\Clean-data\Marawn pablo cleaned.csv')
# marawn_moussa_cleaned=pd.read_csv(r'E:\Projects\Egyptian Rap\Clean-data\Marawn moussa cleaned.csv')
# Wegz_cleaned=pd.read_csv(r'E:\Projects\Egyptian Rap\Clean-data\Wegz cleaned.csv')
# Afroto_cleaned=pd.read_csv(r'E:\Projects\Egyptian Rap\Clean-data\Afroto cleaned.csv')
# abyo_cleaned=pd.read_csv(r'E:\Projects\Egyptian Rap\Clean-data\Abyusif cleaned.csv')

vectorizer = TfidfVectorizer()

options=['Home',"Compare Artists Songs","Artists Words and Info"]
artists=['Marawn pablo','Marawn moussa','Wegz','Afroto','Abyusif',]
selected=option_menu(menu_title=None,options=options,
                     icons=["House","Books"],
                     default_index=0,
                     orientation='horizontal')
def get_similarity_of_songs(artist_df1,artist_df2,song1=None,song2=None,lyrics_col='cleaned_lyrics'):
    vectorizer=TfidfVectorizer()
    text_of_a1=''.join(artist_df1.loc[song1][lyrics_col])
    text_of_a2=''.join(artist_df2.loc[song2][lyrics_col])
    tfidf_matrix=vectorizer.fit_transform([text_of_a1,text_of_a2])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity
def transform_zeros(val):
    if val[0] == 0:
        return 255
    else:
        return val

def display_wordcloud(artist_df,cmap='Reds'):
    mp_text=' '.join(artist_df['cleaned_lyrics'])
    reshaped_text = arabic_reshaper.reshape(mp_text)
    display_text = get_display(reshaped_text)
    word_counts=Counter(mp_text.split())
    font_path=r'Assets\NotoNaskhArabic-VariableFont_wght.ttf'
    # return_obj=wordcloud.visualize(display_text, per_word_coloring=False)
    wordcloud = WordCloud (
                        font_path=font_path).generate(display_text)
    plt.imshow(wordcloud) # image show
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot()

def draw_main_page():
    st.markdown("<h1 style='text-align: center;'>Welcome to TheErapp! 🎧📊🎤.</h1>", unsafe_allow_html=True)
    # Add app description
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        st.markdown("<h2 style='text-align: center;'>Egyption Rap App</h2>", unsafe_allow_html=True)
    with col3:
        st.write(' ')
    st.divider()
    # Add features section
    st.header("Features")
    st.markdown("""<h2>Our text mining app is revolutionizing the way you analyze and compare Egyptian rap songs. 
    With TheErapp, you can easily compare the lyrics of different songs and get detailed text analysis for each artist. 🤖💻🔍</h2>""",unsafe_allow_html=True)
    st.markdown("""<h2>TheErapp uses advanced text mining algorithms to identify similarities between songs and highlight key 
    themes and trends in each artist's lyrics. Whether you're a fan of Marwan Pablo, Wegz, or any other Egyptian rap artist, TheErapp has got you covered. 🔥🎶</h2>""",unsafe_allow_html=True)
def draw_compare_page():
        similarity=0
        st.markdown("<h2 style='text-align:center'> Compare Songs Text similarity See whos the unique artist </h2> 🔍",unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            selection1=st.selectbox('Choose The first artist',artists)
            st.write(' ')
            st.write(' ')
            st.write(' ')
            df1=pd.read_csv(f'Assets\{selection1} cleaned.csv')
            df1.index=df1['Title_english']
            song1=st.selectbox(f'Choose a song from {selection1} list',df1.index,key=1)
            st.image(df1.loc[song1]['image_url'])
        with col2:
            selection2=st.selectbox('Choose The secenod artist',artists[::-1])
            st.write(' ')
            st.write(' ')
            st.write(' ')
            df2=pd.read_csv(f'Assets\{selection2} cleaned.csv')
            df2.index=df2['Title_english']
            song2=st.selectbox(f'Choose a song from {selection2} list',df2.index,key=2)
            st.image(df2.loc[song2]['image_url'])
            st.write(' ')
        if st.button('Compare text'):
            similarity=get_similarity_of_songs(df1,df2,song1,song2)
            st.subheader(f"This two songs are similary by {round(similarity[0][0]*100,2)} %")
def draw_info_page():
        st.markdown("<h2 style='text-align:center'> Information about artist is present here </h2> 🔍",unsafe_allow_html=True)
        selection=st.selectbox('Choose The artist',artists)
        col1, col2,col3,col4 = st.columns(4)
        with col1:
            df1=pd.read_csv(f'Assets\{selection} cleaned.csv')
            df1.index=df1['Title_english']
            st.write(f"<h2>{selection}<h2>",unsafe_allow_html=True)
            if selection == artists[0]:
                st.image(df1.loc['3ayz Fin ']['image_url'],width=600)
            elif selection == artists[1]:
                st.image(df1.loc['1/4 Qarn ']['image_url'],width=600)

            elif selection == artists[2]:
                st.image(df1.loc['ATM ']['image_url'],width=600)

            elif selection == artists[3]:
                st.image(df1.loc['Brazil ']['image_url'],width=600)

            else:
                st.image(df1.loc['2otta ']['image_url'],width=600)
                
        with col2:
            st.write('   ')
        with col3:
            st.metric("Total Number of words",sum(df1['Num_words']))
            st.write(' ')
            st.write(' ')
            st.metric("Average Number of words/song",round(np.mean(df1['Num_words']),2))
            st.title(f'{selection} most common words')
            st.write(' ')
            st.write(' ')
            display_wordcloud(df1)
            with col4:
                st.line_chart(data=df1,x='year',y='Num_words',width=600)
            
        
        
        
        
            
if selected==options[0]:
    draw_main_page()
elif selected==options[1]:
    draw_compare_page()
else:
    draw_info_page()
 

    
