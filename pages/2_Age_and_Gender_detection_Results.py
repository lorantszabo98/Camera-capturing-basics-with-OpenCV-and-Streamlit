import time
import pandas as pd
import streamlit as st
from Data_manager import get_entries_from_session


# @st.cache_data
def load_data(filepath):

    df = pd.read_csv(filepath)
    return df


def save_data(filename, new_dataframe):
    new_dataframe.to_csv(filename, index=False)


def delete_last_row(dataframe):
    if dataframe.last_valid_index() is None:
        status = "error"
        message = "The dataframe is empty!"
        return status, message, dataframe
    else:
        status = "success"
        message = "Last row deleted successfully!"
        new_dataframe = dataframe.drop(dataframe.index[-1])

        return status, message, new_dataframe


st.title("Data Display Page")

loaded_dataframe = load_data("pages/data/data.csv")
new_entries = pd.DataFrame(get_entries_from_session())

final_dataframe = pd.concat([loaded_dataframe, new_entries], ignore_index=True)

dataframe_placeholder = st.dataframe(final_dataframe)

save_data("pages/data/data.csv", final_dataframe)

if st.button("Delete last row"):

    (status, message, new_dataframe) = delete_last_row(final_dataframe)

    if status == "success":

        dataframe_placeholder.dataframe(new_dataframe)
        save_data("pages/data/data.csv", new_dataframe)

        delete_success_message = st.success(message)
        time.sleep(2)
        delete_success_message.empty()

    else:
        delete_error_message = st.error(message)
        time.sleep(2)
        delete_error_message.empty()
