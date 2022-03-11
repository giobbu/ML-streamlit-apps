
import folium
import geopandas as gpd
import json
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_folium import folium_static
import pydeck as pdk
import altair as alt
import matplotlib.pyplot as plt
import pandas as pd

def plot_network(file):
        df_belgium = gpd.read_file(file)
        m = folium.Map([50.85045, 4.34878], zoom_start=9, tiles='cartodbpositron')
        folium.GeoJson(df_belgium).add_to(m)
        return folium_static(m)


def plot_loss(tr_loss, val_loss):
        fig, axes = plt.subplots(1, sharex=True, figsize=(12, 8))
        axes.set_ylabel("Loss (MAE)", fontsize=14)
        axes.plot(tr_loss)
        axes.plot(val_loss)
        axes.set_xlabel("Epoch", fontsize=14)
        leg = axes.legend(loc='upper right')
        return st.pyplot(fig)


def plot_deck(streets):

        INITIAL_VIEW_STATE = pdk.ViewState(latitude=50.85045, longitude=4.34878, zoom=9, max_zoom=9, pitch=45, bearing=0)
        geojson = pdk.Layer(
        "GeoJsonLayer",
        streets['geometry'],
        stroked=False,
        filled=True,
        extruded=True,
        wireframe=True,
        get_elevation= 10,
        get_line_color=[255, 255, 255],
        get_fill_color=[255, 255, 255]
        )

        r = pdk.Deck(layers=geojson, initial_view_state=INITIAL_VIEW_STATE, map_style='mapbox://styles/mapbox/light-v9')
        return r


def plot_line_alt(df):
        df_plot = pd.DataFrame()
        df_line = df.copy()
        df_plot['datetime'] = df_line[['datetime']]
        df_plot['flow_sum'] = df_line.iloc[:,1:].sum(axis=1)
        line = alt.Chart(df_plot).mark_line().encode(x='datetime:T', y='flow_sum:Q', tooltip=['flow_sum', 'datetime']).properties(width=700, height=400, title = ' Total number of OBU trucks').interactive()
        return st.altair_chart(line)


def plot_hist_alt(df):
        df_histogram = pd.DataFrame()
        df_hist = df.copy()
        df_histogram['mean'] = df_hist.mean(axis=0).iloc[1:].values
        arr = alt.Chart(df_histogram).mark_bar().encode(alt.X("mean:Q", bin=alt.Bin(maxbins=300)),y='count()',).properties(width=700, height=400, title = ' OBU traffic mean value distribution on the streets ').interactive()
        return st.altair_chart(arr)


def create_df_stats(df):
        df_stats = pd.DataFrame()
        df_stats = df[['datetime']]
        df_stats['flow_sum'] = df.iloc[:,1:].sum(axis=1)
        df_stats['date'] = pd.to_datetime(df_stats['datetime']).dt.date
        df_stats['week'] = pd.to_datetime(df_stats['datetime']).dt.dayofweek
        df_stats['hour'] = pd.to_datetime(df_stats['datetime']).dt.hour
        return df_stats


def plot_mat_alt(df_max):
        matrix = alt.Chart(df_max).mark_rect().encode(
        alt.X('date(date):O', title='day'),
        alt.Y('month(date):O', title='month'), color='flow_sum:Q').properties(
        title="Daily max OBU traffic flow ").properties(width=700, height=400).interactive()
        return st.altair_chart(matrix)


def plot_trends(df):
        
        df_box_inter = pd.DataFrame()
        df_working = pd.DataFrame()
        df_sat = pd.DataFrame()
        df_sun = pd.DataFrame()

        df_trend = pd.DataFrame()
        df_trend_sat = pd.DataFrame()
        df_trend_sun = pd.DataFrame()
        

        df_box_inter['flow_sum'] = df.iloc[:,1:].sum(axis=1)
        df_box_inter['date'] = pd.to_datetime(df['datetime']).dt.date
        df_box_inter['week'] = pd.to_datetime(df['datetime']).dt.dayofweek
        df_box_inter['time'] = pd.to_datetime(df['datetime']).dt.hour
        df_working = df_box_inter[df_box_inter['week'] < 5]
        df_sat = df_box_inter[df_box_inter['week'] == 5]
        df_sun = df_box_inter[df_box_inter['week'] == 6]

        
        df_trend = df_working.groupby('time').agg({'flow_sum':'mean'}).reset_index()
        df_trend.columns = ['time', 'mean_working_days']

        df_trend_sat = df_sat.groupby('time').agg({'flow_sum':'mean'}).reset_index()
        df_trend_sat.columns = ['time', 'saturday']

        df_trend_sun = df_sun.groupby('time').agg({'flow_sum':'mean'}).reset_index()
        df_trend_sun.columns = ['time', 'sunday']

        df_trend['mean_saturdays'] = df_trend_sat['saturday']
        df_trend['mean_sundays'] = df_trend_sun['sunday']

        trend = alt.Chart(df_trend, title = 'Average daily OBU trends').transform_fold(['mean_working_days', 'mean_saturdays','mean_sundays']).mark_line().encode(x='time:O', y='value:Q', color='key:N').properties(width=700, height=400).interactive()
        return st.altair_chart(trend)



def plot_violin_alt(df_max):

        violin = alt.Chart(df_max, title = 'Max OBU traffic working days/weekend').transform_density(
        'flow_sum',
        as_=['flow_sum', 'density'],
        extent=[0, 400000],
        groupby=['week']
        ).mark_area(orient='horizontal', fillOpacity=0.8).encode(
        y='flow_sum:Q',
        color='week:N',
        x=alt.X(
                'density:Q',
                stack='center',
                impute=None,
                title=None,
                axis=alt.Axis(labels=False, values=[0], grid=True, ticks=False),
        ),
        column=alt.Column(
                'week:N',
                header=alt.Header(
                titleOrient='bottom',
                labelOrient='bottom',
                labelPadding=0,
                ),
        )
        ).properties(
        width=200
        ).configure_facet(
        spacing=10
        ).configure_view(
        stroke=None
        ).interactive()

        return st.altair_chart(violin)


def plot_ridge_alt(df_max, title):

        step = 30
        overlap = 0

        df_max.columns = ['period','flow_sum']

        ridge = alt.Chart(df_max, height=step, width=700).transform_joinaggregate(
        mean_temp='mean(flow_sum)', groupby=['period']
        ).transform_bin(
        ['bin_max', 'bin_min'], 'flow_sum'
        ).transform_aggregate(
        value='count()', groupby=['period', 'flow_sum', 'bin_min', 'bin_max']
        ).transform_impute(
        impute='value', groupby=['period', 'flow_sum'], key='bin_min', value=0
        ).mark_area(
        interpolate='monotone',
        fillOpacity=0.8,
        stroke='lightgray',
        strokeWidth=1.5
        ).encode(
        alt.X('bin_min:Q', bin='binned'), #
        alt.Y(
                'value:Q',
                scale=alt.Scale(range=[step, -step * overlap]),
                axis=None
        ),
        alt.Fill(
                'flow_sum:Q',
                legend=None,
                scale=alt.Scale(scheme='lightmulti')
        )
        ).facet(
        row=alt.Row(
                'period:O',
                title=None,
                # header=alt.Header(labelAngle=0, labelAlign='right', format='%B')
        )
        ).properties(
        title= title).configure_facet(
        spacing=0
        ).configure_title(
        anchor='end'
        ).interactive()

        return st.altair_chart(ridge)