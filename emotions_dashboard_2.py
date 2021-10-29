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
- [2 components solution](#header-1-pca)
- [3 components solution](#header-2-pca)

""")

st.write("""['back to the top'](#toc)]""")

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

st.header('2 components solution', 'header-1-pca')

col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("Forced-choice")
with col2:
    st.subheader("Free-labeling")

image = Image.open('data/pca_chart_2d_images.png')
st.image(image)

image = Image.open('data/pca_chart_2d_text.png')
st.image(image)

st.write("""['back to the top'](#toc)""")

st.header('3 components solution', 'header-2-pca')

st.write("**Interactive charts**: user the pointer to rotate and explore labels")


#######################
## forced-choice PCA 3D
 
df_pca_forced = pd.read_csv('data/forced_choice_pca_3.csv')

fig_forced = px.scatter_3d(df_pca_forced, width=700, height=600, x='x_pca_3', y='y_pca_3', z='z_pca_3',
              color='label')

fig_forced.update_traces(marker=dict(size=7,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

#######################
## free-labeling PCA 3D

df_pca_free = pd.read_csv('data/free_labeling_pca_3.csv')

fig_free = px.scatter_3d(df_pca_free, width=700, height=600, x='x_pca_3', y='y_pca_3', z='z_pca_3',
              color='label')

fig_free.update_traces(marker=dict(size=7,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("Forced-choice")
    st.plotly_chart(fig_forced)
with col2:
    st.subheader("Free-labeling")
    st.plotly_chart(fig_free)

st.write("""['back to the top'](#toc)""")

