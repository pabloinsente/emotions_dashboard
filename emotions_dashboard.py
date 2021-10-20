import streamlit as st
import pandas as pd
import altair as alt
alt.renderers.enable('svg')
alt.themes.enable('vox')
st.set_page_config(layout="wide")

st.write("""
# Emotion categorization surveys results
""")

df = pd.read_csv('clean_data/forced_choice_emotion_uw_students.csv')
df_labels = pd.read_csv('data/emotion_labels.csv')

def count_freq_labels(df, X="all" ):
    if X == "all":
        df_counts = df.stack().reset_index(drop=True).value_counts() # stack as series
        df_counts = df_counts.to_frame('counts') # get value_counts as df
        df_counts['emotion'] = df_counts.index # get index as col
    else:
        df_counts = df[X].reset_index(drop=True).value_counts() # stack as series
        df_counts = df_counts.to_frame('counts') # get value_counts as df
        df_counts[X] = df_counts.index # get index as col

    df_counts = df_counts.reset_index(drop=True) # clean index
    df_counts['percent'] = df_counts['counts'] / df_counts['counts'].sum() # compute percentage
    return df_counts

def simple_per_bar(
    df, title='Title', X='percent:Q', Y='emotion:N', \
    width=450, height=250, sort='-x', \
    text_size = 12, label_size = 11, title_size=12, \
    emotion='Some', color1='#0570b0', color2='orange'):
    
    bars = alt.Chart(df, title=title).mark_bar().encode(
        alt.X(X, axis=alt.Axis(format='.0%')),
        y=alt.Y(Y, sort=sort), 
        color=alt.condition(
            alt.datum.emotion == emotion,
            alt.value(color2),
            alt.value(color1)
        ))
    
    text = bars.mark_text(
    align='left',
    baseline='middle',
    dx=3,  # Nudges text to right so it doesn't appear on top of the bar
    fontSize=text_size,
    fontWeight="bold"
    ).encode(
        alt.Text(X, format='.1%')
    )
    
    chart = (bars + text).configure_axis(
            labelFontSize=label_size,
            titleFontSize=title_size).properties(
                width=width, 
                height=height)
    
    
    return chart

def simple_count_bar(
    df, title='Title', X='counts:Q', Y='emotion:N', \
    width=450, height=250, sort='-x', \
    text_size = 12, label_size = 11, title_size=12,
    emotion='Some', color1='#0570b0', color2='#orange'):
    
    bars = alt.Chart(df, title=title).mark_bar().encode(
        alt.X(X),
        y=alt.Y(Y, sort=sort), 
        color=alt.condition(
            alt.datum.emotion == emotion,
            alt.value(color2),
            alt.value(color1)
        ))
    
    text = bars.mark_text(
    align='left',
    baseline='middle',
    dx=3,  # Nudges text to right so it doesn't appear on top of the bar
    fontSize=text_size,
    fontWeight="bold"
    ).encode(
        alt.Text(X)
    )
    
    chart = (bars + text).configure_axis(
            labelFontSize=label_size,
            titleFontSize=title_size).properties(
                width=width, 
                height=height)
    
    
    return chart

st.write("""
## Demographics
""")

#####################
### - sex chart - ###

source = count_freq_labels(df, X="sex") 
title = 'Sex | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'sex:N'

chart_sex = simple_per_bar(source, title=title, X=X, Y=Y)

#####################
### - age chart - ###

source = count_freq_labels(df, X="age") 
title = 'Age | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'age:N'

chart_age = simple_per_bar(source, title=title, X=X, Y=Y)

###########################
### - ethnicity chart - ###

source = count_freq_labels(df, X="ethnicity") 
title = 'Ethnicity | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'ethnicity:N'

chart_ethnicity= simple_per_bar(source, title=title, X=X, Y=Y)

###########################
### - education chart - ###

source = count_freq_labels(df, X="formal education") 
title = 'Formal education | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'formal education:N'

chart_formal_education= simple_per_bar(source, title=title, X=X, Y=Y)

##########################
### DEMOGRAPHICS BLOCK ###

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    st.subheader("Participants by sex")
    col1.altair_chart(chart_sex, use_container_width=True)
with col2:
    st.subheader("Participants by age group")
    col2.altair_chart(chart_age, use_container_width=True)
with col3:
    st.subheader("Participants by ethnicity")
    col3.altair_chart(chart_ethnicity, use_container_width=True)
with col4:
    st.subheader("Participants by formal education")
    col4.altair_chart(chart_formal_education, use_container_width=True)


st.write("""
## Overall results
""")

df_emo_answers = df.loc[:, 'Q2.1':'Q195.1'] # subset photos


##############################
### - overall per chart - ###

source = count_freq_labels(df_emo_answers, X="all") 
title = 'Labels frequency | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'emotion:N'

chart_overall_per = simple_per_bar(source, title=title, X=X, Y=Y)


##############################
### - overall cnt chart - ###

source = count_freq_labels(df_emo_answers, X="all") 
title = 'Labels frequency | n = '+ source['counts'].sum().astype(str)
X, Y = 'counts:Q', 'emotion:N'

chart_overall_count = simple_count_bar(source, title=title, X=X, Y=Y)

#############################
### OVERALL RESULTS BLOCK ###

col1, col2 = st.columns(2)

with col1:
    st.subheader("Selected labels for all images as percentage")
    col1.altair_chart(chart_overall_per, use_container_width=True)

with col2:
    st.subheader("Selected labels for all images as count")
    col2.altair_chart(chart_overall_count, use_container_width=True)

st.write("""
## Results by expected emotion label
""")

def emotion_df_formated(df_emo_answers, emotion_label):
    df_emo_cat = df_emo_answers.copy() 
    df_emo_cat_t = df_emo_cat.T # transpose
    df_emo_cat_t['photo_id'] = df_emo_cat_t.index # get index as col
    df_emo_cat_t = df_emo_cat_t.reset_index(drop=True) # clean index
    df_emo_cat_t_labels = pd.concat([df_emo_cat_t, df_labels], axis=1) # add metadata cols
    df_label =  df_emo_cat_t_labels[df_emo_cat_t_labels['label'] == emotion_label]
    return df_label

########################
### - anger chart - ###

emotion = 'anger'
df_formated = emotion_df_formated(df_emo_answers, emotion) # subset 'anger' rows
df_formated_ans = df_formated.drop(['photo_id', 'ethnicity', 'sex','age', 'label', 'url'], axis=1)
source = count_freq_labels(df_formated_ans, X="all") 
title = 'Expcted label: '+ emotion + ' | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'emotion:N'

chart_anger = simple_per_bar(source, title=title, X=X, Y=Y, emotion=emotion.capitalize())

#########################
### - disgust chart - ###

emotion = 'disgust'
df_formated = emotion_df_formated(df_emo_answers, emotion) # subset 'anger' rows
df_formated_ans = df_formated.drop(['photo_id', 'ethnicity', 'sex','age', 'label', 'url'], axis=1)
source = count_freq_labels(df_formated_ans, X="all") 
title = 'Expcted label: '+ emotion + ' | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'emotion:N'

chart_disgust = simple_per_bar(source, title=title, X=X, Y=Y, emotion=emotion.capitalize())

######################
### - fear chart - ###

emotion = 'fear'
df_formated = emotion_df_formated(df_emo_answers, emotion) # subset 'anger' rows
df_formated_ans = df_formated.drop(['photo_id', 'ethnicity', 'sex','age', 'label', 'url'], axis=1)
source = count_freq_labels(df_formated_ans, X="all") 
title = 'Expcted label: '+ emotion + ' | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'emotion:N'

chart_fear = simple_per_bar(source, title=title, X=X, Y=Y,  emotion=emotion.capitalize())

#########################
### - surprise chart- ###

emotion = 'surprise'
df_formated = emotion_df_formated(df_emo_answers, emotion) # subset 'anger' rows
df_formated_ans = df_formated.drop(['photo_id', 'ethnicity', 'sex','age', 'label', 'url'], axis=1)
source = count_freq_labels(df_formated_ans, X="all") 
title = 'Expcted label: '+ emotion + ' | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'emotion:N'
w, h= 450, 200
txs, ls, ts = 12, 11, 12

chart_surprise = simple_per_bar(source, title=title, X=X, Y=Y, emotion=emotion.capitalize())

##########################
### - happiness chart- ###

emotion = 'happiness'
df_formated = emotion_df_formated(df_emo_answers, emotion) # subset 'anger' rows
df_formated_ans = df_formated.drop(['photo_id', 'ethnicity', 'sex','age', 'label', 'url'], axis=1)
source = count_freq_labels(df_formated_ans, X="all") 
title = 'Expcted label: '+ emotion + ' | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'emotion:N'

chart_happiness = simple_per_bar(source, title=title, X=X, Y=Y, emotion=emotion.capitalize())   

#########################
### - sadness chart - ###

emotion = 'sadness'
df_formated = emotion_df_formated(df_emo_answers, emotion) # subset 'anger' rows
df_formated_ans = df_formated.drop(['photo_id', 'ethnicity', 'sex','age', 'label', 'url'], axis=1)
source = count_freq_labels(df_formated_ans, X="all") 
title = 'Expcted label: '+ emotion + ' | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'emotion:N'


chart_sadness = simple_per_bar(source, title=title, X=X, Y=Y, emotion=emotion.capitalize())

###########################
### - uncertain chart - ###

emotion = 'uncertain'
df_formated = emotion_df_formated(df_emo_answers, emotion) # subset 'anger' rows
df_formated_ans = df_formated.drop(['photo_id', 'ethnicity', 'sex','age', 'label', 'url'], axis=1)
source = count_freq_labels(df_formated_ans, X="all") 
title = 'Expcted label: '+ emotion + ' | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'emotion:N'

chart_uncertain = simple_per_bar(source, title=title, X=X, Y=Y, emotion=emotion.capitalize())

#########################
### - neutral chart - ###

emotion = 'neutral'
df_formated = emotion_df_formated(df_emo_answers, emotion) # subset 'anger' rows
df_formated_ans = df_formated.drop(['photo_id', 'ethnicity', 'sex','age', 'label', 'url'], axis=1)
source = count_freq_labels(df_formated_ans, X="all") 
title = 'Expcted label: '+ emotion + ' | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'emotion:N'
w, h= 450, 200
txs, ls, ts = 12, 11, 12

chart_neutral = simple_per_bar(source, title=title, X=X, Y=Y, emotion=emotion.capitalize())

########################
### BY EMOTION BLOCK ###

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col5, col6 = st.columns(2)
col7, col8 = st.columns(2)

with col1:
    st.subheader("Selected labels for images depicting 'anger'")
    col1.altair_chart(chart_anger, use_container_width=True)

with col2:
    st.subheader("Selected labels for images depicting 'disgust'")
    col2.altair_chart(chart_disgust, use_container_width=True)

with col3:
    st.subheader("Selected labels for images depicting 'fear'")
    col3.altair_chart(chart_fear, use_container_width=True)

with col4:
    st.subheader("Selected labels for images depicting 'surprise'")
    col4.altair_chart(chart_surprise, use_container_width=True)

with col5:
    st.subheader("Selected labels for images depicting 'happiness'")
    col5.altair_chart(chart_happiness, use_container_width=True)

with col6:
    st.subheader("Selected labels for images depicting 'sadness'")
    col6.altair_chart(chart_sadness, use_container_width=True)

with col7:
    st.subheader("Selected labels for images depicting 'uncertain'")
    col7.altair_chart(chart_uncertain, use_container_width=True)

with col8:
    st.subheader("Selected labels for images depicting 'neutral'")
    col8.altair_chart(chart_neutral, use_container_width=True)

st.write("""
## Results by expected emotion label, and photo's ethnicity
""")

def emotion_df_formated_et(df_emo_answers, emotion_label, ethnicity):
    df_emo_cat = df_emo_answers.copy() 
    df_emo_cat_t = df_emo_cat.T # transpose
    df_emo_cat_t['photo_id'] = df_emo_cat_t.index # get index as col
    df_emo_cat_t = df_emo_cat_t.reset_index(drop=True) # clean index
    df_emo_cat_t_labels = pd.concat([df_emo_cat_t, df_labels], axis=1) # add metadata cols
    df_label =  df_emo_cat_t_labels[(df_emo_cat_t_labels['label'] == emotion_label) & (df_emo_cat_t_labels['ethnicity'] == ethnicity)]
    return df_label

def wrapper_chart_emotion(df_emo_answers, emotion, ethnicity):
    df = emotion_df_formated_et(df_emo_answers, emotion,  ethnicity) # subset 'anger' rows
    df_formated_ans = df.drop(['photo_id', 'ethnicity', 'sex','age', 'label', 'url'], axis=1)
    df_count = count_freq_labels(df_formated_ans) # count label freq
    chart = simple_per_bar(
        df_count, \
        title='Expected label: '+ emotion + ' | n = '+ df_count['counts'].sum().astype(str), \
         emotion=emotion.capitalize())
    return chart

emotion_option = count_freq_labels(df_emo_answers, X="all")['emotion'].tolist() 
emotion_select = st.selectbox("Select emotion", emotion_option)

chart_bipoc = wrapper_chart_emotion(df_emo_answers, emotion_select.lower(), 'bipoc')
chart_white = wrapper_chart_emotion(df_emo_answers, emotion_select.lower(), 'white')

####################################
### BY EMOTION & ETHNICITY BLOCK ###

col1, col2 = st.columns(2)

with col1:
    st.subheader("Selected labels for images depicting BIPOC")
    col1.altair_chart(chart_bipoc, use_container_width=True)
with col2:
    st.subheader("Selected labels for images depicting Caucasian people")
    col2.altair_chart(chart_white, use_container_width=True)
