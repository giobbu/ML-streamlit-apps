from altair.vegalite.v4.schema.channels import Opacity, StrokeDash
import numpy as np
import time
import pandas as pd
import gc
from model.model_module import naive
from util import logging
import tensorflow as tf
from testing.util import inverse_transform, evaluation_fct

import streamlit as st
import altair as alt
from testing.app_util import deck, layer_deck, plot_multistep_error, plot_line_all
import pydeck as pdk


def testing(model, tensor_test, aux_dim, scaler, out_sqc, lst, streets, timestamp, X_tr, Y_tr, X_vl, Y_vl, X_ts, Y_ts):

    logging.info('Testing started')
    forecasts_nv = []
    forecasts = []
    targets = []
    target_plot = []

    nv_rmse_list = []
    nv_mae_list = []

    rmse_list = []
    mae_list = []

    
    st.markdown("""---""")
    st.title('Multivariate Multi-Horizon Traffic Prediction')

    st.header('Prediction from t+1 to t+12 ')
    timestamp_t_h = st.empty()

    st.markdown("""---""")
    st.header('Map Displaying Prediction t+1')
    timestamp_t_1 = st.empty()


    r, INITIAL_VIEW_STATE = layer_deck()
    map = st.pydeck_chart(r)

    st.markdown("""---""")
    
    title =st.empty()


    
    chart_all = st.empty()
    chart_multi = st.empty()

    st.markdown("""---""")
    st.header('Individual Model Performance ')


    st.write('DeepLearn: Historic (red) and current (blue)')
    col3, col4 = st.beta_columns(2)
    with col3:
            st.write('RMSE')
            chart_errorrmse_multi = st.empty()                           
    with col4:
            st.write('MAE')
            chart_errormae_multi = st.empty()



    st.write('NaiveLearn: Historic (red) and current (orange)')
    col5, col6 = st.beta_columns(2)
    with col5:
            st.write('RMSE')
            chart_errorrmse_multi_nv = st.empty()                            
    with col6:
            st.write('MAE')
            chart_errormae_multi_nv = st.empty()


    st.subheader('Model Comparison - DeepLearn (Blue) NaiveLearn (Orange)')

    st.write('Mean Error and StDev for Each Forecast Horizon')
    col7, col8 = st.beta_columns(2)
    with col7:
            st.write('RMSE')
            chart_comp_rmse = st.empty()                            
    with col8:
            st.write('MAE')
            chart_comp_mae = st.empty()

    st.write(' Total Mean Error Tracked Over Time  ')
    st.write('RMSE')
    df_rmse = pd.DataFrame({'timestamp':[],'RMSE_nv':[], 'RMSE_dl': []})
    c = alt.Chart(df_rmse).transform_fold(['RMSE_nv', 'RMSE_dl']).mark_line().encode(x ='timestamp:T', y='value:Q', color='key:N').properties(width=500, height=200)
    chart_rmse = st.altair_chart(c, use_container_width=True)

    st.write('MAE')
    df_mae = pd.DataFrame({'timestamp':[],'MAE_nv': [],'MAE_dl': []})
    c = alt.Chart(df_mae).transform_fold(['MAE_nv', 'MAE_dl']).mark_line().encode(x ='timestamp:T', y='value:Q', color='key:N').properties(width=500, height=200)
    chart_mae = st.altair_chart(c, use_container_width=True)
            
    X_old = X_vl.copy()
    X_new = X_ts.copy()       
    
    for (step, (inp_tot, targ)) in enumerate(tensor_test):

            # naive model
            new_instance = X_new[step]
            X_old = np.insert(X_old, X_old.shape[0], new_instance, axis=0)[1:] 
            pred_nv = naive(X_old, 168*2*2*2, out_sqc).astype(np.int32)
            forecasts_nv.append(pred_nv)
            
            # deep learning model
            inp, aux = inp_tot[0], inp_tot[1]
            targ = tf.cast(targ, tf.float32)

            pred = model(inp, aux, training=False)

            past = inverse_transform(inp[0][:,:- aux_dim],  scaler)
            truth = inverse_transform(targ[0][:,:- aux_dim],  scaler)
            pred = inverse_transform(pred[0][:,:-aux_dim],  scaler)

            forecasts.append(pred)
            targets.append(truth)

            timestamp_t_h.subheader('From '+str(timestamp.iloc[step+12]) +' To ' +str(timestamp.iloc[step+23]))
            
            timestamp_t_1.subheader(str(timestamp.iloc[step+12]))
            r = deck(INITIAL_VIEW_STATE, lst, streets, pred)
            map.pydeck_chart(r)
                      
            rmse_nv, mae_nv = evaluation_fct(targets, forecasts_nv, out_sqc)
            rmse, mae = evaluation_fct(targets, forecasts, out_sqc)

            nv_rmse_list.append(rmse_nv)
            nv_mae_list.append(mae_nv)
            
            rmse_list.append(rmse)
            mae_list.append(mae)

            if step ==0 :
                target_plot.append(past)

                mean_rmse_nv = np.mean(rmse_nv)
                maen_mae_nv = np.mean(mae_nv)

                mean_rmse = np.mean(rmse)
                mean_mae = np.mean(mae)

                rmse_recent, mae_recent= evaluation_fct(targets, forecasts, out_sqc)
                rmse_recent_nv, mae_recent_nv = evaluation_fct(targets, forecasts_nv, out_sqc)

            else:
                target_plot.append(past[-1])

                mean_rmse_nv = np.mean(nv_rmse_list[-1])
                maen_mae_nv = np.mean(nv_mae_list[-1])

                mean_rmse = np.mean(rmse_list[-1])
                mean_mae = np.mean(mae_list[-1])

                rmse_recent, mae_recent = evaluation_fct(list([targets[-1]]), list([forecasts[-1]]), out_sqc)
                rmse_recent_nv, mae_recent_nv = evaluation_fct(list([targets[-1]]), list([forecasts_nv[-1]]), out_sqc)


            mean_pred_multi_nv = np.sum(pred_nv, axis=1)
            mean_pred_multi = np.sum(pred, axis=1)
            mean_truth_multi = np.sum(truth, axis=1)
            all_truth = np.sum(np.vstack(target_plot), axis=1)

            mean_rmse_multi_nv = np.mean(rmse_nv, axis=1)
            mean_stdrmse_multi_nv = np.std(rmse_nv, axis=1) 
            recent_mean_rmse_multi_nv = np.mean(rmse_recent_nv, axis=1)
            recent_mean_stdrmse_multi_nv = np.std(rmse_recent_nv, axis=1)         

            mean_rmse_multi = np.mean(rmse, axis=1)
            mean_stdrmse_multi = np.std(rmse, axis=1)
            recent_mean_rmse_multi = np.mean(rmse_recent, axis=1)
            recent_mean_stdrmse_multi = np.std(rmse_recent, axis=1)

            mean_mae_multi_nv = np.mean(mae_nv, axis=1)
            mean_stdmae_multi_nv = np.std(mae_nv, axis=1)
            recent_mean_mae_multi_nv = np.mean(mae_recent_nv, axis=1)
            recent_mean_stdmae_multi_nv = np.std(mae_recent_nv, axis=1)

            mean_mae_multi = np.mean(mae, axis=1)
            mean_stdmae_multi = np.std(mae, axis=1)
            recent_mean_mae_multi = np.mean(mae_recent, axis=1)
            recent_mean_stdmae_multi = np.std(mae_recent, axis=1)

            time_window = timestamp.iloc[step+12:step+24]

            
            with col3:
                
                rmse_ci, rmse_dot = plot_multistep_error( time_window, mean_rmse_multi, mean_stdrmse_multi, 'red', 0.1 ,  350, 200)
                recent_rmse_ci, recent_rmse_dot = plot_multistep_error( time_window, recent_mean_rmse_multi, recent_mean_stdrmse_multi, 'blue', 0.8 , 350, 200, 'ErrorBar')
                chart_errorrmse_multi.altair_chart( rmse_ci + rmse_dot + recent_rmse_ci + recent_rmse_dot, use_container_width=True)

                rmse_ci, rmse_dot = plot_multistep_error( time_window, mean_rmse_multi_nv, mean_stdrmse_multi_nv, 'red', 0.1 ,  350, 200)
                recent_rmse_ci, recent_rmse_dot = plot_multistep_error( time_window, recent_mean_rmse_multi_nv, recent_mean_stdrmse_multi_nv, 'orange', 0.8 , 350, 200, 'ErrorBar')
                chart_errorrmse_multi_nv.altair_chart( rmse_ci + rmse_dot + recent_rmse_ci + recent_rmse_dot, use_container_width=True)

                rmse_ci_nv, rmse_dot_nv = plot_multistep_error( time_window, recent_mean_rmse_multi_nv, recent_mean_stdrmse_multi_nv, 'orange', 0.2 ,  350, 200)
                rmse_ci, rmse_dot = plot_multistep_error( time_window, recent_mean_rmse_multi, recent_mean_stdrmse_multi, 'blue', 0.1 , 350, 200)
                chart_comp_rmse.altair_chart( rmse_ci + rmse_dot + rmse_ci_nv + rmse_dot_nv, use_container_width=True)

                

            with col4:

                mae_ci, mae_dot = plot_multistep_error( time_window, mean_mae_multi, mean_stdmae_multi, 'red', 0.1 , 350, 200)
                recent_mae_ci, recent_mae_dot = plot_multistep_error( time_window, recent_mean_mae_multi, recent_mean_stdmae_multi, 'blue', 0.8 ,  350, 200, 'ErrorBar')
                chart_errormae_multi.altair_chart( mae_ci + mae_dot + recent_mae_ci + recent_mae_dot, use_container_width=True)

                mae_ci, mae_dot = plot_multistep_error( time_window, mean_mae_multi_nv, mean_stdmae_multi_nv, 'red', 0.1 , 350, 200)
                recent_mae_ci, recent_mae_dot = plot_multistep_error( time_window, recent_mean_mae_multi_nv, recent_mean_stdmae_multi_nv, 'orange', 0.8 ,  350, 200, 'ErrorBar')
                chart_errormae_multi_nv.altair_chart( mae_ci + mae_dot + recent_mae_ci + recent_mae_dot, use_container_width=True)

                mae_ci, mae_dot = plot_multistep_error( time_window, mean_mae_multi_nv, mean_stdmae_multi_nv, 'orange', 0.2 , 350, 200)
                mae_ci_nv, mae_dot_nv = plot_multistep_error( time_window, mean_mae_multi, mean_stdmae_multi, 'blue', 0.1 ,  350, 200)
                chart_comp_mae.altair_chart( mae_ci + mae_dot + mae_ci_nv + mae_dot_nv, use_container_width=True)


            df_rmse = pd.DataFrame({'timestamp':[timestamp.iloc[step+12]],'RMSE_nv': [mean_rmse_nv] ,'RMSE_dl': [mean_rmse]})
            chart_rmse.add_rows(df_rmse)

            df_mae = pd.DataFrame({'timestamp':[timestamp.iloc[step+12]], 'MAE_nv': [maen_mae_nv], 'MAE_dl': [mean_mae]})
            chart_mae.add_rows(df_mae)


            time_past = timestamp.iloc[:step+12]
            time_window = timestamp.iloc[step+12:step+24]

            title.header('Total Belgian Traffic Flow')
            line_past, line_targ, line_pred,line_pred_nv, line_zoom = plot_line_all( time_past, time_window, all_truth, mean_pred_multi, mean_truth_multi, mean_pred_multi_nv, 800, 500)

            chart_all.altair_chart(line_past + line_targ + line_pred + line_pred_nv)
            chart_multi.altair_chart(line_zoom)

            del rmse_ci_nv, rmse_dot_nv, rmse_ci, rmse_dot, recent_rmse_ci, recent_rmse_dot
            del mae_ci_nv, mae_dot_nv, mae_ci, mae_dot, recent_mae_ci, recent_mae_dot
            del line_past, line_targ, line_pred, line_pred_nv, line_zoom
            gc.collect()
            

            time.sleep(5)


    return forecasts, targets, rmse_list, mae_list