#######################
# Imports
#######################

from pandas.tseries.offsets import DateOffset
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from pmdarima import auto_arima
import numpy as np
pio.renderers.default = 'browser'
import pandas as pd
from sqlalchemy import create_engine
import statsmodels.api as sm
from pandas.tseries.offsets import DateOffset

# make any grid with a function
def make_grid(cols,rows):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid

# set size of streamlit page
st.set_page_config(layout="wide")


# set style of dataframe
styles = [

dict(selector="th", props=[("color", "#FFFFFF"),
("border", "1px solid #eee"),
("padding", "12px 35px"),
("border-collapse", "collapse"),
("background", "#002366"),
("text-transform", "uppercase"),
("font-size", "18px")
]),
dict(selector="td", props=[("color", "#FFFFFF"),
("border", "1px solid #eee"),
("padding", "12px 35px"),
("border-collapse", "collapse"),
("font-size", "18px"),
("background", "#002366")
]),
dict(selector="table", props=[
("font-family" , 'Arial'),
("margin" , "25px auto"),
("border-collapse" , "collapse"),
("border" , "1px solid #eee"),
("border-bottom" , "2px solid #00cccc"),
]),
dict(selector="caption", props=[("caption-side", "bottom")])
]




#######################
# Connection with DB
#######################

db_config = {
        'user': 'report',  # имя пользователя
        'pwd': 'DFSeew53dfgxz_dsffh6769675D',  # пароль
        'host': '51.250.69.136',
        'port': 5433,  # порт подключения
        'db': 'wb_dwh'  # название базы данных
        }

connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(
        db_config['user'],
        db_config['pwd'],
        db_config['host'],
        db_config['port'],
        db_config['db'],
    )

engine = create_engine(connection_string)

#######################
# Loading orders
#######################

query = ''' SELECT * FROM stg.orders'''
orders = pd.read_sql_query(query, con=engine)
orders['last_change_date'] = orders['last_change_date'].dt.date
#######################
# Loading stocks
#######################

query = '''SELECT * FROM stg.stocks'''
stocks = pd.read_sql_query(query, con=engine)
stocks['lastchangedate'] = stocks['lastchangedate'].dt.date
#######################
# Creating the buttons
#######################

button_container = st.container()
with button_container:
    number_1 = st.sidebar.number_input('Insert Excess Stock Quantity (days)',
                             min_value=10,
                             max_value=1000,
                             step=1)

    number_2 = st.sidebar.number_input('Insert Insufficient Stock Quantity (days)',
                                       min_value=5,
                                       max_value=1000,
                                       step=1)





#######################
# Stock information
#######################
stock_container = st.container()
with stock_container:
    st.markdown('# Stock general info')
    grid_1 = make_grid(2, (3, 1))

    # Getting up-to-date info about stock
    current_stock_date = stocks.sort_values(by=['lastchangedate'], ascending=False)['lastchangedate'].iloc[0]

    # General info about all stock
    grid_1[0][0].markdown(f'**Articles in stock {current_stock_date}**')
    current_stock = stocks.query('lastchangedate == @current_stock_date')
    fig = go.Figure()
    for warehouse in current_stock['warehousename'].unique():

        stock_by_articles = current_stock.query('warehousename == @warehouse').groupby('nmid')['quantity'].sum().reset_index().sort_values(by = 'quantity', ascending = False)
        exist_stock_by_article = stock_by_articles.query('quantity >0')
        fig.add_trace(go.Bar(x=exist_stock_by_article['nmid'], y=exist_stock_by_article['quantity'], name = warehouse))
    fig.update_layout(barmode='stack', width=900, height=400)
    fig.update_traces(opacity=0.75)
    fig.update_layout(xaxis_type='category', xaxis={'categoryorder': "total descending"})

    grid_1[0][0].plotly_chart(fig)


    grid_1[1][0].markdown(f'**Product groups in stock {current_stock_date}**')
    fig = go.Figure()
    for warehouse in current_stock['warehousename'].unique():
        stock_by_subject = current_stock.query('warehousename == @warehouse').groupby('subject')['quantity'].sum().reset_index().sort_values(by='quantity',
                                                                                                  ascending=False)
        exist_stock_by_subject = stock_by_subject.query('quantity >0')

        fig.add_trace(go.Bar(x=exist_stock_by_subject['subject'], y=exist_stock_by_subject['quantity'], name = warehouse))
    fig.update_layout(barmode='stack', width=900, height=400)
    fig.update_traces(opacity=0.75)
    fig.update_layout(xaxis_type='category', xaxis={'categoryorder': "total descending"})
    grid_1[1][0].plotly_chart(fig)

    # Run out of stock
    grid_1[0][1].markdown('**Articles out of stock**')
    stock_by_articles_cat = current_stock.groupby(['nmid', 'subject'])['quantity'].sum().reset_index().sort_values(by='quantity', ascending=False)
    critical_in_stock = stock_by_articles_cat.query('quantity == 0').reset_index(drop=True).set_index('nmid')
    grid_1[0][1].dataframe(critical_in_stock['subject'])

    grid_1[1][1].markdown('**Product groups out of stock**')
    stock_by_subject_cat = current_stock.groupby(['subject'])['quantity'].sum().reset_index().sort_values(
        by='quantity', ascending=False)
    critical_subject_in_stock = stock_by_subject_cat.query('quantity == 0').reset_index(drop=True).set_index('subject')
    grid_1[1][1].dataframe(critical_subject_in_stock.index)




prediction_container = st.container()
with prediction_container:
    st.markdown('# Stock in days')
    st.markdown('### Please wait. Loading information may take a few seconds')
    @st.experimental_memo(suppress_st_warning=True)
    def skip_computation():
        orders_ = orders.query('is_cancel == False').groupby(['nm_id', 'last_change_date'])['id'].count().reset_index()
        articles = orders_['nm_id'].unique()
        dict_prediction = {}
        for article in articles:
            article_sales = orders_.query('nm_id == @article').set_index('last_change_date')
            idx = pd.date_range(orders['last_change_date'].min(), orders['last_change_date'].max())
            article_sales = article_sales.reindex(idx)
            article_sales.loc[article_sales['nm_id'].isna(), 'nm_id'] = article
            article_sales = article_sales.fillna(0)
            article_sales.columns = ['article', 'sales']
            article_sales = article_sales[['sales']]
            model = sm.tsa.statespace.SARIMAX(article_sales['sales'], order=(0, 0, 0), seasonal_order=(1, 1, 1, 12))
            results = model.fit()
            future_dates = [article_sales.index[-1] + DateOffset(days=x) for x in range(0, 30)]
            future_dataset_df = pd.DataFrame(index=future_dates[1:], columns=article_sales.columns)
            future_df = pd.concat([article_sales, future_dataset_df])
            future_df['forecast'] = results.predict(start=len(article_sales), end=len(future_df), dynamic=True)
            dict_prediction[article] = future_df['forecast'].sum()
        res = pd.DataFrame(dict_prediction.items(), columns=['nm_id', 'predicted_sales']).astype(int)
        art_stock = current_stock.groupby(['nmid'])['quantity'].sum().reset_index().astype(int)
        res = res.merge(art_stock, right_on = 'nmid', left_on = 'nm_id', how = 'left')[['nm_id', 'predicted_sales', 'quantity']].fillna(0)
        res['stock_in_days'] = res['quantity']/res['predicted_sales']*30
        res['quantity'] = res['quantity'].astype(int)
        res['stock_in_days'] = res['stock_in_days'].round(0)
        res.columns = ['nm_id', 'predicted orders for the next 30 days', 'current stock', 'stock_in_days']
        return res

    st.dataframe(skip_computation(), use_container_width=True)
    st.markdown('**<NA> in stock_in_days means there is no demand for this product**')



bad_articles_container = st.container()
with bad_articles_container:
    st.markdown('# Excess goods')
    if number_1:
        st.table(skip_computation().query('stock_in_days >= @number_1').set_index('nm_id').astype(int).style.set_table_styles(styles))

    st.markdown('# Shortage of goods')
    if number_2:
        st.table(skip_computation().query('stock_in_days <= @number_2').set_index('nm_id').astype(int).style.set_table_styles(styles))




article_details_container = st.container()
with article_details_container:
    st.markdown('# Article details')
    articles = orders.query('is_cancel == False')['nm_id'].unique()
    articles_selection = st.multiselect('Choose a product article (nmid)',
                                        options=articles,
                                        default=articles[0])

    for_string = [str(i) for i in articles_selection]
    string_articles = ' '.join(for_string)
    st.dataframe(skip_computation().query('nm_id == @articles_selection').set_index('nm_id'), use_container_width=True)

    for article in articles_selection:
        orders_ = orders.query('is_cancel == False').groupby(['nm_id', 'last_change_date'])['id'].count().reset_index()
        article_sales = orders_.query('nm_id == @article').set_index('last_change_date')
        idx = pd.date_range(orders['last_change_date'].min(), orders['last_change_date'].max())
        article_sales = article_sales.reindex(idx)
        article_sales.loc[article_sales['nm_id'].isna(), 'nm_id'] = article
        article_sales = article_sales.fillna(0)
        article_sales.columns = ['article', 'sales']
        article_sales = article_sales[['sales']]
        model = sm.tsa.statespace.SARIMAX(article_sales['sales'], order=(0, 0, 0), seasonal_order=(1, 1, 1, 12))
        results = model.fit()
        future_dates = [article_sales.index[-1] + DateOffset(days=x) for x in range(0, 30)]
        future_dataset_df = pd.DataFrame(index=future_dates[1:], columns=article_sales.columns)
        future_df = pd.concat([article_sales, future_dataset_df])
        future_df['forecast'] = results.predict(start=len(article_sales), end=len(future_df), dynamic=True)
        future_df = future_df.reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=future_df['index'], y=future_df['sales'], name=f'orders {article}'))
        fig.add_trace(go.Scatter(x=future_df['index'], y=future_df['forecast'], name=f'predicted orders {article}'))
        fig.update_layout(barmode='overlay', width=1300, height=300)
        fig.update_traces(opacity=0.75)
        st.plotly_chart(fig)
