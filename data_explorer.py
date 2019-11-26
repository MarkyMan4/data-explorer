import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
import altair as alt
from iso_regression import IsoRegressor
from lin_regression import LinRegressor
from poly_regression import PolyRegressor
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

    if len(display_data.columns) >= 2:
        if st.checkbox('Variable relationship (scatter matrix)'):
            dim = len(selected_cols) * 2
            fig = plt.figure(figsize=(dim, dim))
            ax = fig.gca()
            scatter_matrix(display_data, ax=ax)
            st.write(fig)

    if ('lat' in selected_cols) and ('lon' in selected_cols):
        if st.checkbox('Show on map'):
            st.map(display_data)

    if st.checkbox('Regression'):
        reg_select = st.selectbox('Select a type of regression', ['Linear', 'Isotonic', 'Polynomial'])

        if reg_select == 'Linear':
            x = st.radio('X', selected_cols)
            y = st.radio('Y', selected_cols)

            lin_reg = LinRegressor(display_data)
            fig = lin_reg.get_graph(x, y)
            st.write(fig)

            input = st.text_input(f'Enter a value for {x} to predict {y}')

            if st.button('Predict'):
                pred = lin_reg.make_prediction(float(input))
                pred = pred[0] # need to get the first value since the prediction is returned as a list
                st.write(f'Predicted {y} = {round(pred, 3)}')
        elif reg_select == 'Isotonic':
            x = st.radio('X', selected_cols)
            y = st.radio('Y', selected_cols)

            ir = IsoRegressor(display_data)
            fig = ir.get_graph(x, y)
            st.write(fig)

            input = st.text_input(f'Enter a value for {x} to predict {y}')

            if st.button('Predict'):
                pred = ir.make_prediction(float(input))
                pred = pred[0] # need to get the first value since the prediction is returned as a list
                st.write(f'Predicted {y} = {round(pred, 3)}')
        elif reg_select == 'Polynomial':
            x = st.radio('X', selected_cols)
            y = st.radio('Y', selected_cols)

            poly_reg = PolyRegressor(display_data)
            fig = poly_reg.get_graph(x, y)
            st.write(fig)

            input = st.text_input(f'Enter a value for {x} to predict {y}')

            if st.button('Predict'):
                pred = poly_reg.make_prediction(float(input))
                pred = pred[0][0] # need to get the first value since the prediction is returned as a list
                st.write(f'Predicted {y} = {round(pred, 3)}')
else:
    st.markdown('## Select at least one column')
