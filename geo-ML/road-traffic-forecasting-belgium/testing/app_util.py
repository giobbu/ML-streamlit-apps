import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
import pydeck as pdk
import yaml
import altair as alt


def layer_deck():
        INITIAL_VIEW_STATE = pdk.ViewState(latitude=50.85045, longitude=4.34878, zoom=8, max_zoom=8, pitch=45, bearing=0)
        r = pdk.Deck(initial_view_state=INITIAL_VIEW_STATE, map_style='mapbox://styles/mapbox/light-v9')
        return r, INITIAL_VIEW_STATE

def deck(INITIAL_VIEW_STATE, lst, streets, pred):

        STREETS = [int(float(s)) for s in lst]
        df = streets[streets.index.isin(STREETS)]
        df['flow'] =  pd.DataFrame(pred).loc[0].astype(float).values

        geojson = pdk.Layer(
                "GeoJsonLayer",
                df,
                stroked=False,
                filled=True,
                extruded=True,
                wireframe=True,
                get_elevation = "flow*30",
                get_fill_color='[255, (1-flow/250)*255, 0]',
                get_line_color='[255, 255, 255]',
                pickable=True
                )

        r = pdk.Deck(layers=[geojson], 
                         initial_view_state=INITIAL_VIEW_STATE,
                         map_style='mapbox://styles/mapbox/light-v9',
                         tooltip={"text": "{flow}"})

        return r



def plot_line_all(time_past, time_window, past, pred, targ, pred_nv, w, h):
            
            df_all = pd.DataFrame({'timestamp': time_past, 'Past_Obs': past})
            line_past = alt.Chart(df_all).transform_fold(['Past_Obs']).mark_area(line={'color':'darkgreen'},color=alt.Gradient(
                    gradient='linear',
                    stops=[alt.GradientStop(color='white', offset=0),
                    alt.GradientStop(color='darkgreen', offset=1)], x1=1, x2=1, y1=1, y2=0)).encode(x='timestamp:T', y='value:Q').properties(width=w, height=h)

            df_multi_targ = pd.DataFrame({'timestamp':time_window, 'Targ': targ})
            line_targ = alt.Chart(df_multi_targ).transform_fold(['Targ']).mark_line(strokeDash=[3,3]).encode(x='timestamp:T', y='value:Q', color= 'key:N').properties(width=w, height=h)
            
            df_multi_pred = pd.DataFrame({'timestamp':time_window, 'Pred': pred})
            line_pred = alt.Chart(df_multi_pred).transform_fold(['Pred']).mark_line().encode(x='timestamp:T', y='value:Q',  color='key:N').properties(width=w, height=h)

            df_multi_pred_nv = pd.DataFrame({'timestamp':time_window, 'Pred_Naive': pred_nv})
            line_pred_nv = alt.Chart(df_multi_pred_nv).transform_fold(['Pred_Naive']).mark_line().encode(x='timestamp:T', y='value:Q',  color='key:N').properties(width=w, height=h)

            df_zoom = pd.DataFrame({'timestamp':time_window, 'Pred': pred, 'Targ': targ})
            line_zoom = alt.Chart(df_zoom, title = 'Zoom on Prediction').transform_fold(['Pred', 'Targ']).mark_line().encode(x='timestamp:T', y='value:Q',color='key:N', strokeDash=alt.condition(alt.datum.value =='Targ', alt.value([5, 5]), alt.value([0]))).properties(width=w, height=h)
            
            return line_past, line_targ, line_pred, line_pred_nv, line_zoom


def plot_multistep_error(time_window, rmse_multi, std_rmse_multi, c, o,  w, h, string = None): #, metric=None):

        if string == 'ErrorBar':
                df_multi_rmse = pd.DataFrame({'timestamp':time_window, 'ymean': rmse_multi, 'ymax': rmse_multi + std_rmse_multi, 'ymin': rmse_multi - std_rmse_multi})
                dot_multi_rmse = alt.Chart(df_multi_rmse).transform_fold(['ymean']).mark_point(opacity = 0.5, color= c).encode(x='timestamp:T', y='ymean:Q').properties(width=w, height=h)
                error = dot_multi_rmse.mark_errorbar(opacity = o, color= c).encode( x="timestamp:T", y="ymin:Q", y2="ymax:Q")
        else:
                df_multi_rmse = pd.DataFrame({'timestamp':time_window, 'ymean': rmse_multi})
                df_multi_rmse_std = pd.DataFrame({'timestamp':time_window, 'ymax': rmse_multi + std_rmse_multi, 'ymin': rmse_multi - std_rmse_multi})

                dot_multi_rmse = alt.Chart(df_multi_rmse).transform_fold(['ymean']).mark_point(opacity = 0.5, color= c).encode(x='timestamp:T', y='ymean:Q').properties(width=w, height=h)
                error = alt.Chart(df_multi_rmse_std).mark_area(opacity = o,  color=c).encode(x='timestamp:T', y='ymax:Q', y2='ymin:Q').properties(width=w, height=h)
          
        return error, dot_multi_rmse 



with open('config.yaml') as file:
        config = yaml.safe_load(file)

path = config['script_path']

