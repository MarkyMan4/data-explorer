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

mode = st.sidebar.selectbox('Explore data or modify it', ['Explore', 'Transform'])

files = os.listdir('data_files')

options = []
for i in range(len(files)):
    if mode == 'Explore' or not files[i].startswith('transformed_'):
        options.append(files[i])

st.sidebar.markdown('# Select data set')
file_select = st.sidebar.selectbox('Add a file to the data_files directory for it to show on this list', options)

data = load_data(file_select)
if mode == 'Explore':
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

            if st.checkbox('Find best model'):
                x = st.radio('X', selected_cols)
                y = st.radio('Y', selected_cols)

                lin_reg = LinRegressor(display_data)
                mae = lin_reg.get_mean_abs_err(x, y)

                st.write(f'Linear regression MAE: {mae}')
    else:
        st.markdown('## Select at least one column')
elif mode == 'Transform':
    st.markdown('### All transformations will be applied to a copy of the file. Your original file will remain intact.')

    if st.checkbox('Show raw data'):
        st.write(data)

    transformation = st.selectbox('Select a transformation', ['-select-', 'Apply change to column'])

    if transformation == 'Apply change to column':
        columns_to_change = []

        for i in range(len(data.columns)):
            columns_to_change.append(st.checkbox(data.columns[i]))

        # only use columns selected in check boxes
        selected_cols = []
        for i in range(len(columns_to_change)):
            if columns_to_change[i]:
                selected_cols.append(data.columns[i])

        column_change_options = ['Add', 'Sub', 'Mul', 'Div']
        selected_change = st.selectbox('Select change to make', column_change_options)

        change_function = lambda x : x

        if selected_change == 'Add':
            amt_to_add = st.number_input('Value to add')
            change_function = lambda x : x + amt_to_add
        elif selected_change == 'Sub':
            amt_to_sub = st.number_input('Value to subtract')
            change_function = lambda x : x - amt_to_sub
        elif selected_change == 'Mul':
            mult_by = st.number_input('Value to multiply by')
            change_function = lambda x : x * mult_by
        elif selected_change == 'Div':
            div_by = st.number_input('Value to divide by')
            change_function = lambda x : x / div_by

        if st.button('Apply'):
            transformed_data = data.copy()

            for col in selected_cols:
                transformed_data[col] = transformed_data[col].apply(change_function)

            transformed_file_name = 'transformed_' + file_select
            transformed_data.to_csv('data_files/' + transformed_file_name, index=False)
