import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import base64
import plotly.express as px
from PIL import Image


st.set_page_config(layout="wide")

st.title('Emotion categorization analysis', 'top')

st.write("""

My dissertation investigates three questions at the intersection of cognitive science and the psychology of emotion:  
 
(1) how humans **categorize** and **conceptualize** facial expressions of emotion  
(2) whether different **methods** for asking people to categorize facial expressions alters such behavior  
(3) whether perceived **social characteristics** of faces (like race and gender) impacts humans categorization behavior  

I approached these questions in three studies on three different samples. I'll briefly describe the stimulus and samples studied, 
to then concisely describe the three studies.

## Images of facial expressions 

I study categorization behavior on images of facial expressions, as it is the only source of stimulus that satisfies the requirements 
for my research goals. These requirements are:

(1) Images from a wide **variety of ethnic, nationality, sex, and age groups**   
(2) Images obtained in a **"naturalistic-like" manner**. This contrasts to images obtained in studio settings, from paid actors 
*posing* an exaggerated facial configuration of what is thought in western culture as archetypical of certain emotion

The [AffecNet database](http://mohammadmahoor.com/affectnet/) of facial expressions "in the wild" satisfy these criteria. It contains 
over 1 million of facial images gathered from the internet, by querying 1250 emotion related terms in six languages. Around 440 thousand 
of these images were manually annotated by trained raters, following the seven so-called ["basic" or "universal" emotions](https://www.paulekman.com/resources/universal-facial-expressions/): 
 *anger, disgust, fear, happiness, neutral, sadness, and surprises*. 

 Utilizing the annotated images from the AffecNet database, it is not an endorsement of such a model, but an opportunity to test whether
 using this conceptualization model holds when confronted to different methodologies for collecting judgments about facial expressions.

## Populations under study

As this research project was done during the COVID-19 pandemic, all the data collection was completed on-line. Data was collected from 
three groups:

(1) Undergraduates from a large public university in the United States  
(2) English-speaking adults workers from the United States in Amazon MTurk   
(3) Spanish-speaking adults workers from South America in Amazon MTurk (pending...)  

Sample sizes were approximately 100 participants (on each group) for the first study, and between 200-300 participants for the second 
study, yielding a total of approximately 1100 participants overall.

## Study I: Comparing forced-choice and free-choice methods for the categorization of facial expressions of emotion

The first study, consist of comparing two methods to measure how humans categorize facial expressions of emotion:  

(1) A **"forced-choice" method**, where participants are presented with a series of images of facial expressions, and asked
 to select among the seven alternatives provided by the basic emotions model. This is the most widely used method in the literature, 
 and the one used by the trained raters to categorize the AffecNet images    
(2) A **"free-choice" (or free-labeling) method**, where participants are presented with the same images, but allowed to produce 
their own categories to label the images

## Study II: A dueling-bandit approach to rank descriptors for facial expressions of emotion 

When asking people to categorize facial expressions, we are effectively asking them to *rank* potential descriptors of such expressions. 
The force-choice method requires participants to pick the best descriptor among seven pre-selected alternatives, whereas the free-choice 
method requires selecting the best descriptor among the ones that comes to mind while looking at the image. A reasonable question is 
whether people would select a different term to describe a facial expression, if they have access to a wider range of words, either written 
down in from of them, or in the lexicon of their minds during that particular instance in time. They may have chosen differently. 

To address the above-mentioned issue, I utilize a **dueling-banding method** approach, where participants are presented, sequentially, with
pairs of words to describe a facial expression, and then select which word (from that pair) is the best descriptor. Dueling-bandits experiments 
have been used as a ranking method for a variety of problems in the past, but not, to my knowledge, to rank descriptors for facial expressions 
of emotion. 

The words selected for this study, are the ones produced by participants in the free-choice survey from my first study, instead of being 
manually selected by myself. Concretely, I combined all the words used to describe a face belonging to a given emotion (in accordance to 
the annotations in AffecNet), and use that set to run the dueling-bandit study. 

This study was implemented in NEXT, a platform that leverages active learning to crowdsourcing. NEXT computes [Borda Scores](https://en.wikipedia.org/wiki/Borda_count), 
wich then used to rank descriptors for each face image. 

## Study III: Cultural variations in the categorization of facial expression of emotion

Language and culture defines how we process the world around us, particularly or social interactions. Whether categorization of facial 
expressions of emotion it is also subject to these forces, it has been a contested subject in the literature. A coarse generalization 
of the debate is the following: the "universalist", like Paul Ekman and colleagues, who claim that emotions are categorical entities 
universally recognizable across cultures, versus the "constructivist", like James A. Russell and Lisa F. Barret, who claim that emotions 
are contextual and socioculturally fabricated entities. We can add to these contrasting views a plethora of authors "in between", 
proposing variations or combinations of both intellectual streams.  

In my third study I contribute to disentangle the debate, by comparing the results obtained in Study I and II, to a sample of native 
Spanish-speaking participants from South America, which happens to be the region of the world where I come from. 

""")

st.header('Navigation', 'navigation')

st.write("""

This is a long dashboard. To jump to the results for each sample of participants, click on the links below:

- **I. Undergraduate students sample results.** [Jump to section](#st-sample)
- **II. English-speaking MTurk sample results.** [Jump to section](#mturk-sample)
- **III. Snpanish-speaking MTurk sample results.** [Jump to section](#mturk-sample-spanish)

""")

st.title('I: Undergraduate students sample results', 'st-sample')

st.subheader('Table of contents', 'toc')

st.write("""
[**Forced-choice results**](#title-fc):
- [Participants demographics](#header-fc-dem)
- [Overall results](#header-fc-overall)
- [Results by emotion label](#header-fc-emotions)
- [Results by emotion label and ethnicity](#header-fc-emotions-et)

[**Free-labeling results**](#title-fl):
- [Participants demographics](#header-fl-dem)
- [Overall results](#header-fl-overall)
- [Results by emotion label](#header-fl-emotions)
- [Results by emotion label and ethnicity](#header-fl-emotions-et)

[**K-means clustering results**](#title-km):
- [Forced-choice clustering](#header-fc-km)
    - [K-means cluster selection criteria](#subheader-fc-km-e)
    - [K-means solution](#subheader-fc-km-s)
- [Free-labeling clustering](#header-fl-km)
    - [K-means cluster selection criteria](#subheader-fl-km-e)
    - [K-means solution](#subheader-fl-km-s)
- [K-means clustering evaluation](#header-km-eval)
    - [Silhouette score](#subheader-km-ss)
    - [Calinski-Harabasz score](#subheader-km-chs)
    - [Davies-Bouldin score](#subheader-km-dbs)

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

[**Dueling-bandits ranking experiment and comparison with surveys results**](#title-db):
- [Participants demographics](#header-db-dem)
- [Word-ranking for 'anger'](#header-db-anger)
- [Word-ranking for 'disgust'](#header-db-disgust)
- [Word-ranking for 'fear'](#header-db-fear)
- [Word-ranking for 'happiness'](#header-db-happiness)
- [Word-ranking for 'neutral'](#header-db-neutral)
- [Word-ranking for 'sadness'](#header-db-sadness)
- [Word-ranking for 'surprise'](#header-db-surprise)

""")

st.write("""[back to the toc](#toc)""")

## SVG images ##
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

st.write("""[back to the toc](#toc)""")

st.header('Overall results', 'header-fc-overall')

## overall % ##
st.write(df_svg['image_title'][4])
render_svg(df_svg['svg'][4])

## overall cnt ##
st.write(df_svg['image_title'][5])
render_svg(df_svg['svg'][5])

st.write("""[back to the toc](#toc)""")

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

st.write("""[back to the toc](#toc)""")

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

st.write("""[back to the toc](#toc)""")

st.title('Free-labeling results', 'title-fl')
df_svg_free = pd.read_csv('data/free_choice_svg_strings.csv')

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

st.write("""[back to the toc](#toc)""")


st.header('Overall results', 'header-fl-overall')

## overall % ##
st.write(df_svg_free['image_title'][4])
render_svg(df_svg_free['svg'][4])

## overall cnt ##
st.write(df_svg_free['image_title'][5])
render_svg(df_svg_free['svg'][5])

st.write("""[back to the toc](#toc)""")

st.header('Results by expected emotion label', 'header-fl-emotions')

## anger ##
st.write(df_svg_free['image_title'][6])
render_svg(df_svg_free['svg'][6])

## disgust  ##
st.write(df_svg_free['image_title'][7])
render_svg(df_svg_free['svg'][7])

## fear  ##
st.write(df_svg_free['image_title'][8])
render_svg(df_svg_free['svg'][8])

## surprise ##
st.write(df_svg_free['image_title'][9])
render_svg(df_svg_free['svg'][9])

## happiness  ##
st.write(df_svg_free['image_title'][10])
render_svg(df_svg_free['svg'][10])

## sadness ##
st.write(df_svg_free['image_title'][11])
render_svg(df_svg_free['svg'][11])

## uncertain  ##
st.write(df_svg_free['image_title'][12])
render_svg(df_svg_free['svg'][12])

## neutral ##
st.write(df_svg_free['image_title'][13])
render_svg(df_svg_free['svg'][13])

st.write("""[back to the toc](#toc)""")

st.header('Results by expected emotion and ethnicity', 'header-fl-emotions-et')

col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("POC images")
with col2:
    st.subheader("Caucasian images")

## anger * ethnicity ##
st.write(df_svg_free['image_title'][14])
render_svg(df_svg_free['svg'][14])

## disgust * ethnicity ##
st.write(df_svg_free['image_title'][15])
render_svg(df_svg_free['svg'][15])

## fear * ethnicity ##
st.write(df_svg_free['image_title'][16])
render_svg(df_svg_free['svg'][16])

## surprise * ethnicity ##
st.write(df_svg_free['image_title'][17])
render_svg(df_svg_free['svg'][17])

## happiness * ethnicity ##
st.write(df_svg_free['image_title'][18])
render_svg(df_svg_free['svg'][18])

## sadness * ethnicity ##
st.write(df_svg_free['image_title'][19])
render_svg(df_svg_free['svg'][19])

## neutral * ethnicity ##
st.write(df_svg_free['image_title'][20])
render_svg(df_svg_free['svg'][20])

## uncertain * ethnicity ##
st.write(df_svg_free['image_title'][21])
render_svg(df_svg_free['svg'][21])

st.write("""[back to the toc](#toc)""")


##################
## K MEANS RESULTS

st.title('K-means clustering results', 'title-km')

###########################
## Forced choice clustering

st.header('Forced-choice clustering', 'header-fc-km')

## K-means *evaluation* forced choice ##
st.subheader(df_svg['image_title'][22], 'subheader-fc-km-e')
render_svg(df_svg['svg'][22])

## K-means *clusters* forced choice ##

st.subheader('K-means 6 clusters solution', 'subheader-fc-km-s')
image = Image.open('data/k_means_forced_choice_6.png')
st.image(image)

st.write("""[back to the toc](#toc)""")

###########################
## Free labeling clustering

st.header('Free-labeling clustering', 'header-fl-km')

## K-means *evaluation* free choice ##
st.subheader(df_svg_free['image_title'][22], 'subheader-fl-km-e')
render_svg(df_svg_free['svg'][22])

## K-means *clusters* forced choice ##

st.subheader('K-means 10 clusters solution', 'subheader-fl-km-s')
image = Image.open('data/k_means_free_labeling_10.png')
st.image(image)

st.write("""[back to the toc](#toc)""")

##############################
## KMeans clusters evaluation
###############################

df_pca_eval = pd.read_csv('data/pca_svg_strings.csv')

st.header('K-means clustering evaluation', 'header-km-eval')

### silhouette_score ###

st.subheader(df_pca_eval['image_title'][0])
render_svg(df_pca_eval['svg'][0])

### calinski_harabasz_score ###

st.subheader(df_pca_eval['image_title'][1])
render_svg(df_pca_eval['svg'][1])

### davies_bouldin_score ###

st.subheader(df_pca_eval['image_title'][2])
render_svg(df_pca_eval['svg'][2])

st.write("""[back to the toc](#toc)""")


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

st.write("""[back to the toc](#toc)""")


## forced-choice by ethnicity ##

st.header('Forced-choice 2 components solution by image ethnicity', 'header-pca-2d-forced-et')

col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("POC")
with col2:
    st.subheader("Caucasian")

image = Image.open('data/pca_chart_2d_images_forced_ethnicity.png')
st.image(image)

image = Image.open('data/pca_chart_2d_text_forced_ethnicity.png')
st.image(image)

st.write("""[back to the toc](#toc)""")

## free-choice by ethnicity ##

st.header('Free-labeling 2 components solution by image ethnicity', 'header-pca-2d-free-et')

col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("POC")
with col2:
    st.subheader("Caucasian")

image = Image.open('data/pca_chart_2d_images_free_ethnicity.png')
st.image(image)

image = Image.open('data/pca_chart_2d_text_free_ethnicity.png')
st.image(image)

st.write("""[back to the toc](#toc)""")

#####################
## PCA evaluation ##
####################

df_pca_eval = pd.read_csv('data/pca_svg_strings.csv')

st.header('PCA 2 components embeddings evaluation', 'header-2d-pca-eval')

### silhouette_score ###

st.subheader(df_pca_eval['image_title'][3])
render_svg(df_pca_eval['svg'][3])

### calinski_harabasz_score ###

st.subheader(df_pca_eval['image_title'][4])
render_svg(df_pca_eval['svg'][4])

### davies_bouldin_score ###

st.subheader(df_pca_eval['image_title'][5])
render_svg(df_pca_eval['svg'][5])

st.write("""[back to the toc](#toc)""")


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
        xaxis = dict(nticks=8, range=[-1.7,1.7],),
        yaxis = dict(nticks=8, range=[-1.7,1.7],),
        zaxis = dict(nticks=8, range=[-1.7,1.7],)))


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
        xaxis = dict(nticks=8, range=[-1.7,1.7],),
        yaxis = dict(nticks=8, range=[-1.7,1.7],),
        zaxis = dict(nticks=8, range=[-1.7,1.7],)))


st.header('3 components solution by survey method', 'header-pca-3d-all')

# PLOT #
col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("Forced-choice")
    st.plotly_chart(fig_forced)
with col2:
    st.subheader("Free-labeling")
    st.plotly_chart(fig_free)

st.write("""[back to the toc](#toc)""")


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
        xaxis = dict(nticks=8, range=[-1.7,1.7],),
        yaxis = dict(nticks=8, range=[-1.7,1.7],),
        zaxis = dict(nticks=8, range=[-1.7,1.7],)))

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
        xaxis = dict(nticks=8, range=[-1.7,1.7],),
        yaxis = dict(nticks=8, range=[-1.7,2.2],),
        zaxis = dict(nticks=8, range=[-1.7,1.7],)))

st.header('3 components solution by survey method - BIPOC', 'header-pca-3d-bipoc')

# PLOT #
col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("Forced-choice - BIPOC")
    st.plotly_chart(fig_forced)
with col2:
    st.subheader("Free-labeling - BIPOC")
    st.plotly_chart(fig_free)

st.write("""[back to the toc](#toc)""")


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
        xaxis = dict(nticks=8, range=[-1.7,1.7],),
        yaxis = dict(nticks=8, range=[-1.7,1.7],),
        zaxis = dict(nticks=8, range=[-1.7,1.7],)))


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
        xaxis = dict(nticks=8, range=[-1.7,2.2],),
        yaxis = dict(nticks=8, range=[-1.7,1.7],),
        zaxis = dict(nticks=8, range=[-1.7,1.7],)))


st.header('3 components solution by survey method - Caucasian', 'header-pca-3d-caucasian')

# PLOT #
col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("Forced-choice - Caucasian")
    st.plotly_chart(fig_forced)
with col2:
    st.subheader("Free-labeling - Caucasian")
    st.plotly_chart(fig_free)

st.write("""[back to the toc](#toc)""")


## PCA ****3D**** evaluation ##

st.header('PCA 3 components embeddings evaluation', 'header-3d-pca-eval')

### silhouette_score ###

st.subheader(df_pca_eval['image_title'][6])
render_svg(df_pca_eval['svg'][6])

### calinski_harabasz_score ###

st.subheader(df_pca_eval['image_title'][7])
render_svg(df_pca_eval['svg'][7])

### davies_bouldin_score ###

st.subheader(df_pca_eval['image_title'][8])
render_svg(df_pca_eval['svg'][8])

st.write("""[back to the toc](#toc)""")

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

st.write("""[back to the toc](#toc)""")

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

st.write("""[back to the toc](#toc)""")

#############################
## DUELING BANDINTS EXPERIMENT

st.title('Dueling-bandits ranking experiment and comparison with surveys results', 'title-db')

## DEMOGRAPHICS ##
st.header('Participants demographics', 'header-db-dem')
st.write("""[back to the toc](#toc)""")

## ANGER RANKINGS##
st.header('Word-rankings for anger', 'header-db-anger')

# anger female of color #
image = Image.open('data/anger_bfa_next_panel.png')
st.image(image)

# anger male of color #
image = Image.open('data/anger_bma_next_panel.png')
st.image(image)

# anger white female #
image = Image.open('data/anger_wfa_next_panel.png')
st.image(image)

# anger white male  #
image = Image.open('data/anger_wma_next_panel.png')
st.image(image)

st.write("""[back to the toc](#toc)""")

## DISGUST RANKINGS ##
st.header('Word-rankings for disgust', 'header-db-disgust')

# disgust female of color #
image = Image.open('data/disgust_bfa_next_panel.png')
st.image(image)

# disgust male of color #
image = Image.open('data/disgust_bma_next_panel.png')
st.image(image)

# disgust white female #
image = Image.open('data/disgust_wfa_next_panel.png')
st.image(image)

# disgust white male  #
image = Image.open('data/disgust_wma_next_panel.png')
st.image(image)

## FEAR RANKINGS ##
st.header('Word-rankings for fear', 'header-db-fear')

# fear female of color #
image = Image.open('data/fear_bfa_next_panel.png')
st.image(image)

# fear male of color #
image = Image.open('data/fear_bma_next_panel.png')
st.image(image)

# fear white female #
image = Image.open('data/fear_wfa_next_panel.png')
st.image(image)

# fear white male  #
image = Image.open('data/fear_wma_next_panel.png')
st.image(image)

st.write("""[back to the toc](#toc)""")

## HAPPINESS RANKINGS ##
st.header('Word-rankings for happiness', 'header-db-happiness')

# happiness female of color #
image = Image.open('data/happiness_bfa_next_panel.png')
st.image(image)

# happiness male of color #
image = Image.open('data/happiness_bma_next_panel.png')
st.image(image)

st.write("""[back to the toc](#toc)""")

## NEUTRAL RANKINGS ##
st.header('Word-rankings for neutral', 'header-db-neutral')
st.write("""pending data...""")
st.write("""[back to the toc](#toc)""")

## SADNESS RANKINGS ##
st.header('Word-rankings for sadness', 'header-db-sadness')
st.write("""pending data...""")
st.write("""[back to the toc](#toc)""")

## SURPRISE RANKINGS ##
st.header('Word-rankings for surprise', 'header-db-surprise')
st.write("""pending data...""")
st.write("""[back to the toc](#toc)""")

##########################
##########################
## English-speaking MTurk 
##########################
##########################

st.title('II: English-speaking MTurk sample results', 'mturk-sample')

st.subheader('Table of contents', 'toc-mturk')

st.write("""
[**Forced-choice results**](#title-fc-mturk):
- [Participants demographics](#header-fc-dem-mturk)
- [Overall results](#header-fc-overall-mturk)
- [Results by emotion label](#header-fc-emotions-mturk)
- [Results by emotion label and ethnicity](#header-fc-emotions-et-mturk)

[**Free-labeling results**](#title-fl-mturk):
- [Participants demographics](#header-fl-dem-mturk)
- [Overall results](#header-fl-overall-mturk)
- [Results by emotion label](#header-fl-emotions-mturk)
- [Results by emotion label and ethnicity](#header-fl-emotions-et-mturk)

[**K-means clustering results**](#title-km-mturk):
- [Forced-choice clustering](#header-fc-km-mturk)
    - [K-means cluster selection criteria](#subheader-fc-km-e-mturk)
    - [K-means solution](#subheader-fc-km-s-mturk)
- [Free-labeling clustering](#header-fl-km-mturk)
    - [K-means cluster selection criteria](#subheader-fl-km-e-mturk)
    - [K-means solution](#subheader-fl-km-s-mturk)
- [K-means clustering evaluation](#header-km-eval-mturk)
    - [Silhouette score](#subheader-km-ss-mturk)
    - [Calinski-Harabasz score](#subheader-km-chs-mturk)
    - [Davies-Bouldin score](#subheader-km-dbs-mturk)

[**PCA results**](#title-pca-mturk):
- [2 components solution by survey method](#header-pca-2d-all-mturk)
- [Forced-choice 2 components solution by image ethnicity](#header-pca-2d-forced-et-mturk)
- [Free-labeling 2 components solution by image ethnicity](#header-pca-2d-free-et-mturk)
- [PCA 2 components embeddings evaluation](#header-2d-pca-eval-mturk)
    - [Silhouette score](#subheader-pca-2d-ss-mturk)
    - [Calinski-Harabasz score](#subheader-pca-2d-chs-mturk)
    - [Davies-Bouldin score](#subheader-pca-2d-dbs-mturk)

""")

st.write("""[back to the toc](#toc-mturk)""")

df_svg = pd.read_csv('data/forced_choice_svg_strings_mturk.csv')

st.title('Forced-choice results', 'title-fc-mturk')

st.header('Participants demographics', 'header-fc-dem-mturk')


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

st.write("""[back to the toc](#mturk-sample)""")

st.header('Overall results', 'header-fc-overall-mturk')

## overall % ##
st.write(df_svg['image_title'][4])
render_svg(df_svg['svg'][4])

## overall cnt ##
st.write(df_svg['image_title'][5])
render_svg(df_svg['svg'][5])

st.write("""[back to the toc](#mturk-sample)""")

st.header('Results by expected emotion label', 'header-fc-emotions-mturk')

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

st.write("""[back to the toc](#mturk-sample)""")

st.header('Results by expected emotion and ethnicity', 'header-fc-emotions-et-mturk')

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

st.write("""[back to the toc](#mturk-sample)""")


st.title('Free-labeling results', 'title-fl-mturk')

df_svg_free = pd.read_csv('data/free_choice_svg_strings_mturk.csv')

st.header('Participants demographics', 'header-fl-dem-mturk')

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

st.write("""[back to the toc](#mturk-sample)""")


st.header('Overall results', 'header-fl-overall-mturk')

## overall % ##
st.write(df_svg_free['image_title'][4])
render_svg(df_svg_free['svg'][4])

## overall cnt ##
st.write(df_svg_free['image_title'][5])
render_svg(df_svg_free['svg'][5])

st.write("""[back to the toc](#mturk-sample)""")

st.header('Results by expected emotion label', 'header-fl-emotions-mturk')

## anger ##
st.write(df_svg_free['image_title'][6])
render_svg(df_svg_free['svg'][6])

## disgust  ##
st.write(df_svg_free['image_title'][7])
render_svg(df_svg_free['svg'][7])

## fear  ##
st.write(df_svg_free['image_title'][8])
render_svg(df_svg_free['svg'][8])

## surprise ##
st.write(df_svg_free['image_title'][9])
render_svg(df_svg_free['svg'][9])

## happiness  ##
st.write(df_svg_free['image_title'][10])
render_svg(df_svg_free['svg'][10])

## sadness ##
st.write(df_svg_free['image_title'][11])
render_svg(df_svg_free['svg'][11])

## uncertain  ##
st.write(df_svg_free['image_title'][12])
render_svg(df_svg_free['svg'][12])

## neutral ##
st.write(df_svg_free['image_title'][13])
render_svg(df_svg_free['svg'][13])

st.write("""[back to the toc](#mturk-sample)""")

st.header('Results by expected emotion and ethnicity', 'header-fl-emotions-et-mturk')

col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("POC images")
with col2:
    st.subheader("Caucasian images")

## anger * ethnicity ##
st.write(df_svg_free['image_title'][14])
render_svg(df_svg_free['svg'][14])

## disgust * ethnicity ##
st.write(df_svg_free['image_title'][15])
render_svg(df_svg_free['svg'][15])

## fear * ethnicity ##
st.write(df_svg_free['image_title'][16])
render_svg(df_svg_free['svg'][16])

## surprise * ethnicity ##
st.write(df_svg_free['image_title'][17])
render_svg(df_svg_free['svg'][17])

## happiness * ethnicity ##
st.write(df_svg_free['image_title'][18])
render_svg(df_svg_free['svg'][18])

## sadness * ethnicity ##
st.write(df_svg_free['image_title'][19])
render_svg(df_svg_free['svg'][19])

## neutral * ethnicity ##
st.write(df_svg_free['image_title'][20])
render_svg(df_svg_free['svg'][20])

## uncertain * ethnicity ##
st.write(df_svg_free['image_title'][21])
render_svg(df_svg_free['svg'][21])

st.write("""[back to the toc](#mturk-sample)""")


##################
## K MEANS RESULTS

st.title('K-means clustering results', 'title-km-mturk')

###########################
## Forced choice clustering

st.header('Forced-choice clustering', 'header-fc-km-mturk')

## K-means *evaluation* forced choice ##
st.subheader(df_svg['image_title'][22], 'subheader-fc-km-e-mturk')
render_svg(df_svg['svg'][22])

## K-means *clusters* forced choice ##
st.subheader('K-means 6 clusters solution', 'subheader-fc-km-s-mturk')
image = Image.open('data/k_means_forced_choice_6_mturk.png')
st.image(image)

st.write("""[back to the toc](#mturk-sample)""")

###########################
## Free labeling clustering

st.header('Free-labeling clustering', 'header-fl-km-mturk')

## K-means *evaluation* free choice ##
st.subheader(df_svg_free['image_title'][22], 'subheader-fl-km-e-mturk')
render_svg(df_svg_free['svg'][22])

## K-means *clusters* forced choice ##

st.subheader('K-means 10 clusters solution', 'subheader-fl-km-s-mturk')
image = Image.open('data/k_means_free_choice_10_mturk.png')
st.image(image)

st.write("""[back to the toc](#mturk-sample)""")

##############################
## KMeans clusters evaluation
###############################

df_pca_eval = pd.read_csv('data/pca_svg_strings_mturk.csv')

st.header('K-means clustering evaluation', 'header-km-eval-mturk')

### silhouette_score ###

st.subheader(df_pca_eval['image_title'][0])
render_svg(df_pca_eval['svg'][0])

### calinski_harabasz_score ###

st.subheader(df_pca_eval['image_title'][1])
render_svg(df_pca_eval['svg'][1])

### davies_bouldin_score ###

st.subheader(df_pca_eval['image_title'][2])
render_svg(df_pca_eval['svg'][2])

st.write("""[back to the toc](#mturk-sample)""")

##############
## PCA RESULTS

st.title('PCA results', 'title-pca-mturk')

## aggregated ##

st.header('2 components solution by survey method', 'header-pca-2d-all-mturk')

col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("Forced-choice")
with col2:
    st.subheader("Free-labeling")

image = Image.open('data/pca_chart_2d_images_all_mturk.png')
st.image(image)

image = Image.open('data/pca_chart_2d_text_all_mturk.png')
st.image(image)

st.write("""[back to the toc](#mturk-sample)""")


## forced-choice by ethnicity ##

st.header('Forced-choice 2 components solution by image ethnicity', 'header-pca-2d-forced-et-mturk')

col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("POC")
with col2:
    st.subheader("Caucasian")

image = Image.open('data/pca_chart_2d_images_forced_ethnicity_mturk.png')
st.image(image)

image = Image.open('data/pca_chart_2d_text_forced_ethnicity_mturk.png')
st.image(image)

st.write("""[back to the toc](#mturk-sample)""")

## free-choice by ethnicity ##

st.header('Free-labeling 2 components solution by image ethnicity', 'header-pca-2d-free-et-mturk')

col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("POC")
with col2:
    st.subheader("Caucasian")

image = Image.open('data/pca_chart_2d_images_free_ethnicity_mturk.png')
st.image(image)

image = Image.open('data/pca_chart_2d_text_free_ethnicity_mturk.png')
st.image(image)

st.write("""[back to the toc](#mturk-sample)""")

#####################
## PCA evaluation ##
####################

st.header('PCA 2 components embeddings evaluation', 'header-2d-pca-eval-mturk')

### silhouette_score ###

st.subheader(df_pca_eval['image_title'][3])
render_svg(df_pca_eval['svg'][3])

### calinski_harabasz_score ###

st.subheader(df_pca_eval['image_title'][4])
render_svg(df_pca_eval['svg'][4])

### davies_bouldin_score ###

st.subheader(df_pca_eval['image_title'][5])
render_svg(df_pca_eval['svg'][5])

st.write("""[back to the toc](#mturk-sample)""")

# --------- %% --------- #

##########################
##########################
## Spanish-speaking MTurk 
##########################
##########################

# --------- %% --------- #


st.title('III: Spanish-speaking MTurk sample results', 'mturk-sample-spanish')

st.write('Pending data...')
