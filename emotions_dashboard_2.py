import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import base64
import plotly.express as px


st.set_page_config(layout="wide")

st.title('Emotion categorization surveys results', 'top')

st.header('Table of contents', 'toc')

st.write("""
[**Forced-choice results**](#title-fc):
- [Participants demographics](#header-fc-dem)
- [Overall results](#header-fc-overall)
- [Results by emotion label](#header-fc-emotions)
- [Results by emotion label and ethnicity](#header-fc-emotions-et)

[**Free-labeling results**](#title-fl):
- [Participants demographics](#header-fl-dem)
- ... in progress

[**K-means clustering results**](#title-km):
- [Forced-choice clustering](#header-fc-km)
    - [K-means evaluation](#subheader-fc-km-e)
    - [K-means solution](#subheader-fc-km-s)
- [Free-labeling clustering](#header-fl-km)
    - [K-means evaluation](#subheader-fl-km-e)
    - [K-means solution](#subheader-fl-km-s)

[**PCA results**](#title-pca):
- [2 components solution by survey method](#header-pca-2d-all)
- [Forced-choice 2 components solution by image ethnicity](#header-pca-2d-forced-et)
- [Free-labeling 2 components solution by image ethnicity](#header-pca-2d-free-et)
- [PCA 2 components embeddings evaluation](#header-2d-pca-eval)
    - [Silhouette score](#subheader-pca-2d-ss)
    - [Calinski-Harabasz score](#subheader-pca-2d-chs)
    - [Davies-Bouldin score](#subheader-pca-2d-dbs)
- [3 components solution by survey method](#header-pca-3d-all)
- [3 components solution by survey method - BIPOC](#header-pca-3d-bipoc)
- [3 components solution by survey method - Caucasian](#header-pca-3d-caucasian)
- [PCA 3 components embeddings evaluation](#header-3d-pca-eval)
    - [Silhouette score](#subheader-pca-3d-ss)
    - [Calinski-Harabasz score](#subheader-pca-3d-chs)
    - [Davies-Bouldin score](#subheader-pca-3d-dbs)


[**Sentiment analysis results**](#title-sen):
- [Sentiment score distributions](#header-sen-d)
- [Sentiment score means and confidence intervals](#header-sen-m)

""")

st.write("""['back to the top'](#toc)""")

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)

df_svg = pd.read_csv('data/forced_choice_svg_strings.csv')

st.title('Forced-choice results', 'title-fc')

st.header('Participants demographics', 'header-fc-dem')

## participants by sex ##
st.write(df_svg['image_title'][0])
render_svg(df_svg['svg'][0])

## participants by age ##
st.write(df_svg['image_title'][1])
render_svg(df_svg['svg'][1])

## participants by ethnicity ##
st.write(df_svg['image_title'][2])
render_svg(df_svg['svg'][2])

## participants by formal education ##
st.write(df_svg['image_title'][3])
render_svg(df_svg['svg'][3])

st.write("""['back to the top'](#toc)""")

st.header('Overall results', 'header-fc-overall')

## overall % ##
st.write(df_svg['image_title'][4])
render_svg(df_svg['svg'][4])

## overall cnt ##
st.write(df_svg['image_title'][5])
render_svg(df_svg['svg'][5])

st.write("""['back to the top'](#toc)""")

st.header('Results by expected emotion label', 'header-fc-emotions')

## anger ##
st.write(df_svg['image_title'][6])
render_svg(df_svg['svg'][6])

## disgust  ##
st.write(df_svg['image_title'][7])
render_svg(df_svg['svg'][7])

## fear  ##
st.write(df_svg['image_title'][8])
render_svg(df_svg['svg'][8])

## surprise ##
st.write(df_svg['image_title'][9])
render_svg(df_svg['svg'][9])

## happiness  ##
st.write(df_svg['image_title'][10])
render_svg(df_svg['svg'][10])

## sadness ##
st.write(df_svg['image_title'][11])
render_svg(df_svg['svg'][11])

## uncertain  ##
st.write(df_svg['image_title'][12])
render_svg(df_svg['svg'][12])

## neutral ##
st.write(df_svg['image_title'][13])
render_svg(df_svg['svg'][13])

st.write("""['back to the top'](#toc)""")

st.header('Results by expected emotion and ethnicity', 'header-fc-emotions-et')

col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("POC images")
with col2:
    st.subheader("Caucasian images")

## anger * ethnicity ##
st.write(df_svg['image_title'][14])
render_svg(df_svg['svg'][14])

## disgust * ethnicity ##
st.write(df_svg['image_title'][15])
render_svg(df_svg['svg'][15])

## fear * ethnicity ##
st.write(df_svg['image_title'][16])
render_svg(df_svg['svg'][16])

## surprise * ethnicity ##
st.write(df_svg['image_title'][17])
render_svg(df_svg['svg'][17])

## happiness * ethnicity ##
st.write(df_svg['image_title'][18])
render_svg(df_svg['svg'][18])

## sadness * ethnicity ##
st.write(df_svg['image_title'][19])
render_svg(df_svg['svg'][19])

## neutral * ethnicity ##
st.write(df_svg['image_title'][20])
render_svg(df_svg['svg'][20])

## uncertain * ethnicity ##
st.write(df_svg['image_title'][21])
render_svg(df_svg['svg'][21])

st.write("""['back to the top'](#toc)""")

st.title('Free-labeling results', 'title-fl')
df_svg_free = pd.read_csv('data/free_choice_svg_strings.csv')

st.write("""In progress...""")



st.header('Participants demographics', 'header-fl-dem')

## participants by sex ##
st.write(df_svg_free['image_title'][0])
render_svg(df_svg_free['svg'][0])

## participants by age ##
st.write(df_svg_free['image_title'][1])
render_svg(df_svg_free['svg'][1])

## participants by ethnicity ##
st.write(df_svg_free['image_title'][2])
render_svg(df_svg_free['svg'][2])

## participants by formal education ##
st.write(df_svg_free['image_title'][3])
render_svg(df_svg_free['svg'][3])

st.write("""['back to the top'](#toc)""")



##################
## K MEANS RESULTS

st.title('K-means clustering results', 'title-km')

###########################
## Forced choice clustering

st.header('Forced-choice clustering', 'header-fc-km')

## K-means evaluation forced choice ##
st.subheader(df_svg['image_title'][22], 'subheader-fc-km-e')
render_svg(df_svg['svg'][22])

## K-means clusters forced choice ##
from PIL import Image

st.subheader(df_svg['image_title'][23], 'subheader-fc-km-s')
image = Image.open('data/k_means_forced_choice_6.png')
st.image(image)

st.write("""['back to the top'](#toc)""")

###########################
## Free labeling clustering

st.header('Free-labeling clustering', 'header-fl-km')

## K-means evaluation free choice ##
st.subheader(df_svg_free['image_title'][4], 'subheader-fl-km-e')
render_svg(df_svg_free['svg'][4])

## K-means clusters forced choice ##

st.subheader('K-means 10 clusters solution', 'subheader-fl-km-s')
image = Image.open('data/k_means_free_labeling_10.png')
st.image(image)

st.write("""['back to the top'](#toc)""")

##############
## PCA RESULTS

st.title('PCA results', 'title-pca')

## aggregated ##

st.header('2 components solution by survey method', 'header-pca-2d-all')

col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("Forced-choice")
with col2:
    st.subheader("Free-labeling")

image = Image.open('data/pca_chart_2d_images_all.png')
st.image(image)

image = Image.open('data/pca_chart_2d_text_all.png')
st.image(image)

st.write("""['back to the top'](#toc)""")


## forced-choice by ethnicity ##

st.header('Forced-choice 2 components solution by image ethnicity', 'header-pca-2d-forced-et')

col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("Caucasian")
with col2:
    st.subheader("Bipoc")

image = Image.open('data/pca_chart_2d_images_forced_ethnicity.png')
st.image(image)

image = Image.open('data/pca_chart_2d_text_forced_ethnicity.png')
st.image(image)

st.write("""['back to the top'](#toc)""")

## free-choice by ethnicity ##

st.header('Free-labeling 2 components solution by image ethnicity', 'header-pca-2d-free-et')

col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("Caucasian")
with col2:
    st.subheader("Bipoc")

image = Image.open('data/pca_chart_2d_images_free_ethnicity.png')
st.image(image)

image = Image.open('data/pca_chart_2d_text_free_ethnicity.png')
st.image(image)

st.write("""['back to the top'](#toc)""")


## PCA evaluation ##

df_pca_eval = pd.read_csv('data/pca_svg_strings.csv')

st.header('PCA 2 components embeddings evaluation', 'header-2d-pca-eval')

### silhouette_score ###

st.subheader(df_pca_eval['image_title'][0])
render_svg(df_pca_eval['svg'][0])

### calinski_harabasz_score ###

st.subheader(df_pca_eval['image_title'][1])
render_svg(df_pca_eval['svg'][1])

### davies_bouldin_score ###

st.subheader(df_pca_eval['image_title'][2])
render_svg(df_pca_eval['svg'][2])

st.write("""['back to the top'](#toc)""")


st.header('3 components solution by survey method', 'header-pca-3d-all')

st.write("**Interactive charts**: user the pointer to rotate and explore labels")


#######################
## forced-choice PCA 3D

df_label_a = pd.read_csv('data/pca_3d_aggregated_forced.csv')

fig_forced = px.scatter_3d(df_label_a, width=700, height=600, x='x_pca_3', y='y_pca_3', z='z_pca_3',
              color='label')

fig_forced.update_traces(marker=dict(size=8,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

fig_forced.update_layout(
    scene = dict(
        xaxis = dict(nticks=8, range=[-1.6,1.6],),
        yaxis = dict(nticks=8, range=[-1.6,1.6],),
        zaxis = dict(nticks=8, range=[-1.6,1.6],)))


#######################
## free-labeling PCA 3D

df_label_free_a = pd.read_csv('data/pca_3d_aggregated_free.csv')

fig_free = px.scatter_3d(df_label_free_a, width=700, height=600, x='x_pca_3', y='y_pca_3', z='z_pca_3',
              color='label')

fig_free.update_traces(marker=dict(size=7,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

fig_free.update_layout(
    scene = dict(
        xaxis = dict(nticks=8, range=[-1.6,1.6],),
        yaxis = dict(nticks=8, range=[-1.6,1.6],),
        zaxis = dict(nticks=8, range=[-1.6,1.6],)))


st.header('3 components solution by survey method', 'header-pca-3d-all')

# PLOT #
col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("Forced-choice")
    st.plotly_chart(fig_forced)
with col2:
    st.subheader("Free-labeling")
    st.plotly_chart(fig_free)

st.write("""['back to the top'](#toc)""")


#######################
## forced-choice PCA 3D - BIPOC

df_label_b = pd.read_csv('data/pca_3d_bipoc_forced.csv')

fig_forced = px.scatter_3d(df_label_b, width=700, height=600, x='x_pca_3', y='y_pca_3', z='z_pca_3',
              color='label')

fig_forced.update_traces(marker=dict(size=8,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

fig_forced.update_layout(
    scene = dict(
        xaxis = dict(nticks=8, range=[-1.6,1.6],),
        yaxis = dict(nticks=8, range=[-1.6,1.6],),
        zaxis = dict(nticks=8, range=[-1.6,1.6],)))

#######################
## Free-choice PCA 3D - BIPOC

df_label_free_b = pd.read_csv('data/pca_3d_bipoc_free.csv')

fig_free = px.scatter_3d(df_label_free_b, width=700, height=600, x='x_pca_3', y='y_pca_3', z='z_pca_3',
              color='label')

fig_free.update_traces(marker=dict(size=8,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

fig_free.update_layout(
    scene = dict(
        xaxis = dict(nticks=8, range=[-1.6,1.6],),
        yaxis = dict(nticks=8, range=[-1.6,1.6],),
        zaxis = dict(nticks=8, range=[-1.6,1.6],)))

st.header('3 components solution by survey method - BIPOC', 'header-pca-3d-bipoc')

# PLOT #
col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("Forced-choice - BIPOC")
    st.plotly_chart(fig_forced)
with col2:
    st.subheader("Free-labeling - BIPOC")
    st.plotly_chart(fig_free)

st.write("""['back to the top'](#toc)""")


#######################
## forced-choice PCA 3D - Caucasian

df_label_c = pd.read_csv('data/pca_3d_caucasian_forced.csv')

fig_forced = px.scatter_3d(df_label_b, width=700, height=600, x='x_pca_3', y='y_pca_3', z='z_pca_3',
              color='label')

fig_forced.update_traces(marker=dict(size=8,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

fig_forced.update_layout(
    scene = dict(
        xaxis = dict(nticks=8, range=[-1.6,1.6],),
        yaxis = dict(nticks=8, range=[-1.6,1.6],),
        zaxis = dict(nticks=8, range=[-1.6,1.6],)))


#######################
## Free-choice PCA 3D - Caucasian

df_label_free_c = pd.read_csv('data/pca_3d_caucasian_free.csv')

fig_free = px.scatter_3d(df_label_free_c, width=700, height=600, x='x_pca_3', y='y_pca_3', z='z_pca_3',
              color='label')

fig_free.update_traces(marker=dict(size=8,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

fig_free.update_layout(
    scene = dict(
        xaxis = dict(nticks=8, range=[-1.6,1.6],),
        yaxis = dict(nticks=8, range=[-1.6,1.6],),
        zaxis = dict(nticks=8, range=[-1.6,1.6],)))


st.header('3 components solution by survey method - Caucasian', 'header-pca-3d-caucasian')

# PLOT #
col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("Forced-choice - Caucasian")
    st.plotly_chart(fig_forced)
with col2:
    st.subheader("Free-labeling - Caucasian")
    st.plotly_chart(fig_free)

st.write("""['back to the top'](#toc)""")


## PCA ****3D**** evaluation ##

st.header('PCA 3 components embeddings evaluation', 'header-3d-pca-eval')

### silhouette_score ###

st.subheader(df_pca_eval['image_title'][3])
render_svg(df_pca_eval['svg'][3])

### calinski_harabasz_score ###

st.subheader(df_pca_eval['image_title'][4])
render_svg(df_pca_eval['svg'][4])

### davies_bouldin_score ###

st.subheader(df_pca_eval['image_title'][5])
render_svg(df_pca_eval['svg'][5])

st.write("""['back to the top'](#toc)""")

#####################
## SENTIMENT ANALYSIS

df_sentiment_svg = pd.read_csv('data/sentiment_svg_strings.csv')

st.title('Sentiment analysis results', 'title-sen')

#################################
## sentiment scores distributions

st.header('Sentiment score distributions', 'header-sen-d')

col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("Forced-choice")
with col2:
    st.subheader("Free-labeling")

## overall sentiment ##
st.subheader(df_sentiment_svg['image_title'][0])
render_svg(df_sentiment_svg['svg'][0])

## overall sex ##
st.subheader(df_sentiment_svg['image_title'][1])
render_svg(df_sentiment_svg['svg'][1])

## overall ethnicity ##
st.subheader(df_sentiment_svg['image_title'][2])
render_svg(df_sentiment_svg['svg'][2])

## overall age ##
st.subheader(df_sentiment_svg['image_title'][3])
render_svg(df_sentiment_svg['svg'][3])

st.write("""['back to the top'](#toc)""")

#############################
## sentiment scores means/std

st.header('Sentiment score means and confidence intervals', 'header-sen-m')

col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("Forced-choice")
with col2:
    st.subheader("Free-labeling")

## compounded sentiment ##
st.subheader(df_sentiment_svg['image_title'][4])
render_svg(df_sentiment_svg['svg'][4])

## negative sentiment ##
st.subheader(df_sentiment_svg['image_title'][5])
render_svg(df_sentiment_svg['svg'][5])

## positive sentiment ##
st.subheader(df_sentiment_svg['image_title'][6])
render_svg(df_sentiment_svg['svg'][6])

## neutral sentiment ##
st.subheader(df_sentiment_svg['image_title'][7])
render_svg(df_sentiment_svg['svg'][7])

st.write("""['back to the top'](#toc)""")
