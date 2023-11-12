import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time


def load_data(filepath):

    df = pd.read_csv(filepath)
    return df


def display_data_bar_chart(dataframe, column):
   counts = dataframe[column].value_counts()
   st.bar_chart(counts, use_container_width=True)
   st.caption(f"Bar Chart of {column} Counts")


st.title("Data Display")

tab1, tab2 = st.tabs(["Data", "Chart"])

dataframe = load_data("pages/data/data.csv")

with tab1:
    st.dataframe(dataframe)

with tab2:
    option = add_selectbox = st.selectbox(
        "Which data do you want to display?",
        ("Date", "Gender", "Age"),
        index=2
    )

    if option == "Date":
        display_data_bar_chart(dataframe, "Date")
        # st.balloons()

    if option == "Gender":
        display_data_bar_chart(dataframe, "Gender")

    if option == "Age":
        display_data_bar_chart(dataframe, "Age")
        # st.snow()









