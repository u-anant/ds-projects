import pandas as pd
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

sarima_model = SARIMAXResultsWrapper.load('models/sarima_1001117.pickle')

st.title('Prediction of Energy Consumption')
Ntest = st.slider('Select Forecast Horizon', 14, 30, 14)

if st.button('FORECAST'):
        
    train_pred = sarima_model.fittedvalues
    prediction_result = sarima_model.get_forecast(Ntest)
    conf_int = prediction_result.conf_int()
    lower, upper = conf_int['lower Energy'], conf_int['upper Energy']
    forecast = prediction_result.predicted_mean

    train_pred = pd.DataFrame(data={'Training': train_pred.values}, index=train_pred.index)
    forecast = pd.DataFrame(data={'Forecast': forecast.values}, index=forecast.index)

    df_idx = pd.date_range(start=train_pred.index[0], end=forecast.index[-1])
    final_df = pd.DataFrame(index=df_idx)

    final_df = final_df.join(train_pred)
    final_df = final_df.join(forecast)

    st.header('Plot : Energy Consumption per Day')
    st.line_chart(final_df)
