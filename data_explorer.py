import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
import os

@st.cache
def load_data(filename):
    data = pd.read_csv('data_files/' + filename)
    return data

st.title('Data Explorer')

files = os.listdir('data_files')

options = []
for i in range(len(files)):
    options.append(files[i])

st.sidebar.markdown('# Select a file to explore')
file_select = st.sidebar.selectbox('Add a file to the data_files directory for it to show on this list', options)

data = load_data(file_select)
c_boxes = []

for i in range(len(data.columns)):
    c_boxes.append(st.sidebar.checkbox(data.columns[i]))

st.sidebar.markdown('# ')
st.sidebar.markdown('### See the data:')
if st.sidebar.checkbox('Explore'):

    # only use columns selected in check boxes
    selected_cols = []
    for i in range(len(c_boxes)):
        if c_boxes[i]:
            selected_cols.append(data.columns[i])

    display_data = data[selected_cols]

    if len(selected_cols) > 0:
        if st.checkbox('Show raw data'):
            st.write(display_data)

        if st.checkbox('Line chart'):
            st.line_chart(display_data)

        if st.checkbox('Bar chart'):
            st.bar_chart(display_data)

        if st.checkbox('Variable relationship (scatter matrix)'):
            if len(display_data.columns) < 2:
                st.write('select at least 2 columns to view this')
            else:
                dim = len(display_data.columns) * 2
                fig = plt.figure(figsize=(dim, dim))
                ax = fig.gca()
                scatter_matrix(display_data, ax=ax)
                st.write(fig)
    else:
        st.markdown('## Select at least one column')
else:
     st.markdown('## Click Explore to start')
