
# ML packages
import yaml
import pickle

from util import logging, load_streets, set_seed
from app_util import plot_network, plot_loss, plot_deck
from app_util import plot_line_alt, plot_hist_alt, create_df_stats, plot_mat_alt, plot_trends, plot_violin_alt, plot_ridge_alt

from data.data_module import data_reader, feat_engin, data_split_scale, data_loader, data_loader_naive
from data.data_module import data_loader, data_loader_naive

from data.app_util import plot_distribution, plot_gantt, plot_tensor
from model.model_module import LSTM_ED
from training.train_module import training
from training.util import opt, loss_fct, valid_loss_fct
from training.app_util import slider_display, save_updates_yaml, upload_yaml
from testing.test_module import testing


import time
import os
import numpy as np
import tensorflow as tf
import folium
import geopandas as gpd
import pandas as pd


# App packages
import streamlit as st
import altair as alt
import SessionState
import pydeck as pdk
import matplotlib.pyplot as plt

session_state = SessionState.get(check1=False)

file_streets = 'data/Belgium_streets.json'
checkpoint_dir = 'trained_model_dir'


with open('config.yaml') as file:
        config = yaml.safe_load(file)


path = config['script_path']
mean_value = config['data']['threshold']
n_feat_time = config['data']['time_feature']
validation_period = config['data']['validation']
testing_period = config['data']['testing']

inp_sqc = config['loader']['input_sqc']
out_sqc = config['loader']['output_sqc']
total_dim = config['loader']['tot_dim']
aux_dim = n_feat_time
batch_tr = config['loader']['batch_tr']
batch_vl = config['loader']['batch_tr']
batch_ts =  config['loader']['batch_ts']






def main():

        st.set_page_config(page_title="DeapLearn_NaiveLearn_App")

        set_seed(42)

        st.markdown("""---""")
        # Space out the maps so the first one is 2x the size of the other three
        with st.container():

                col1, col2 = st.columns([20,20])
                
                with col1:
                        st.image('img/mlg.png', width=250)
                with col2:
                        st.image('img/ULB.jpeg', width=200)
                        
                st.markdown("""---""")

                st.title('On-Board-Unit Data: LSTM-Encoder Decoder for Freight Traffic Forecasting')

                st.markdown("""---""")

                st.write('Reliable Traffic Forecasting schemes are fundamental to design Proactive Intelligent Transportation Systems (ITS). In this context deep learning models have recently shown promising results. The Streamlit App presents a tutorial on multi-horizon traffic flow predictions with LSTM encoder-decoder model.')

                st.write("Check the paper [HERE](https://www.researchgate.net/publication/348930068_A_Tutorial_on_Network-Wide_Multi-Horizon_Traffic_Forecasting_with_Deep_Learning)")
                st.image('img/ECDEC.jpg', width=500)
                st.markdown("""---""")


                if st.button('Data Summary'):
                        st.title('Overview')
                        df = data_reader(path)
                        streets = load_streets(file_streets)
                        
                        st.header('Highways')
                        st.text('Folium Visualization')
                        plot_network(file_streets)
                        
                        st.header('Traffic Data')
                        st.subheader('Raw OBU Data')              
                        st.dataframe(df.head())

                        col0, col1, col2 = st.columns(3)

                        with col0:
                                st.subheader('Temporal')
                                st.text('-- Period-- ')
                                st.text('from ' + df.iloc[0,0])
                                st.text('to '+ df.iloc[-1,0])
                                st.text('Granularity: 30 minutes')
                                st.text('Number Observations ' + str(df.shape[0]))

                        with col1:
                                st.subheader('Spatial')
                                st.text('Number of Streets: ' +str(df.shape[1]))
                                st.text('-- Bounds Box --')
                                box = streets.bounds.values[0]
                                st.text('Upper Left: ' + str(round(box[3],3)) +', '+ str(round(box[0],3)))
                                st.text('Lower Right: ' + str(round(box[1],3)) +', '+ str(round(box[2],3)))

                        with col2:
                                st.subheader('Traffic')
                                mean = round(np.mean(df.iloc[:,1:].sum(axis=1).values),0)
                                st.text('Total Mean: ' + str(mean))
                                max = np.max(df.iloc[:,1:].sum(axis=1))
                                st.text('Total Max: ' + str(max))
                                min = np.min(df.iloc[:,1:].sum(axis=1))
                                st.text('Total Min: ' + str(min))

                        st.markdown("""---""")

                        if st.button("BACK"):
                                st.text("Restarting...")

                elif st.button('Data Exploration'):

                        st.markdown("""---""")
                        st.title('Info & Viz')

                        df = data_reader(path)
                        # compute max OBU traffic
                        df_stats = create_df_stats(df)
                        streets = load_streets(file_streets)
                        
                        st.subheader('Check the whole period')
                        # plot timeseries total obu traffic
                        plot_line_alt(df)

                        # plot histogram mean obu traffic
                        plot_hist_alt(df)

                        # plot matrix OBU traffic
                        df_max = df_stats.groupby('date').agg({'flow_sum':'max'}).reset_index()
                        plot_mat_alt(df_max)
                        
                        st.subheader('Check working days/weekend')
                        # plot avg trend working, sat, sund
                        plot_trends(df)

                        # plot max OBU data working, sat, sund
                        df_ww = df_stats.copy()
                        df_ww.loc[df_ww.week < 5, "week"] = "max_working_days"
                        df_ww.loc[df_ww.week == 5, "week"] = "max_saturdays"
                        df_ww.loc[df_ww.week ==6, "week"] = "max_sundays"
                        df_max = df_ww.groupby('week').agg({'flow_sum':'max'}).reset_index()
                        plot_violin_alt(df_max)


                        st.subheader('Check days of the week')

                        # plot max OBU data distribution days of the week

                        df_stats.loc[df_stats.week == 0, "week"] = "1_mon"
                        df_stats.loc[df_stats.week == 1, "week"] = "2_tue"
                        df_stats.loc[df_stats.week == 2, "week"] = "3_wed"
                        df_stats.loc[df_stats.week == 3, "week"] = "4_thu"
                        df_stats.loc[df_stats.week == 4, "week"] = "5_fri"
                        df_stats.loc[df_stats.week == 5, "week"] = "6_sat"
                        df_stats.loc[df_stats.week ==6, "week"] = "7_sun"
                        df_max = df_stats.groupby('week').agg({'flow_sum':'max'}).reset_index()
                        plot_ridge_alt(df_max, title = 'Max OBU traffic per day of the week ')
                        
                        st.subheader('Check hours of working day')
                        df_max = df_stats.groupby('hour').agg({'flow_sum':'max'}).reset_index()
                        plot_ridge_alt(df_max, title = 'Max OBU traffic per hour of the working days ')



                        if st.button("BACK"):
                                st.text("Restarting...")

                elif st.button('Data Preparation'):

                        df = data_reader(path)
                        streets = load_streets(file_streets)                

                        # select meaninful streets (with variance) and perform feature engineering
                        st.subheader('Select Streets')

                        st.text('select streets with avg traffic flow higher than '+str(mean_value)+' number trucks/30 minutes')
                        df_new, lst_streets, timestamp = feat_engin(mean_value, df)
                        
                        plot_distribution(df_new, 'blue', 700, 400)

                        st.markdown("""---""")

                        st.subheader('Add Features')
                        st.text('create temporal covariates to improve model learning')
                        st.text(list(df_new.columns[-4:])[0])
                        st.text(list(df_new.columns[-4:])[1])
                        st.text(list(df_new.columns[-4:])[2])
                        st.text(list(df_new.columns[-4:])[3])


                        st.markdown("""---""")

                        with st.container():
                                
                                st.header('Naive Time Series Split')
                                st.text('1 week of OBU data for validation,  1 week for testing and rest for training the model')
                                # split and scale the data
                                TRAIN, VAL, TEST, train, val, test, scaler, timestamp_test = data_split_scale(df_new, validation_period, testing_period, n_feat_time, timestamp)

                                
                                st.subheader('Line Plot')
                                x = list(range(df_new.shape[0]))
                                line_tr = train[:,:-4].mean(axis=1)
                                line_vl = val[:,:-4].mean(axis=1)
                                line_ts = test[:,:-4].mean(axis=1)
                                labels = ['train','validation','test']

                                fig, ax = plt.subplots()
                                ax.plot(x[:train.shape[0]], line_tr, color='blue')
                                ax.plot(x[train.shape[0]:train.shape[0]+val.shape[0]], line_vl, color='orange')
                                ax.plot(x[train.shape[0]+val.shape[0]:train.shape[0]+val.shape[0]+test.shape[0]], line_ts, color='green')
                                ax.legend(labels)
                                st.pyplot(fig, width =10) 

                                st.subheader('Data sets distributions after scaling (0,1)')
                                col6, col7, col8 = st.columns(3)
                                
                                with col6:               
                                        st.text('Training set')
                                        plot_distribution(train, 'blue', 250, 300)
                                                
                                with col7:
                                        st.text('Validation set')
                                        plot_distribution(val, 'orange', 250, 300)

                                with col8:
                                        st.text('Testing set')
                                        plot_distribution(test, 'green', 250, 300)

                                st.markdown("""---""")

                                st.header('Prepare Tensors for Naive Model ')
                                X_tr, Y_tr, X_vl, Y_vl, X_ts, Y_ts = data_loader_naive(df_new.values[:, : -4], inp_sqc, out_sqc, validation_period, testing_period)

                                with open('naive_datasets.pkl','wb') as f:
                                        pickle.dump([X_tr, Y_tr, X_vl, Y_vl, X_ts, Y_ts], f)


                                st.write(X_vl.shape)
                                st.write(X_ts.shape)

                                st.markdown("""---""")

                                st.header('Prepare Tensors for Deep Learning - (Batch, Sequence, Features)')

                                st.markdown("""---""")

                                st.text('Input sequence length : ' +str(inp_sqc))
                                st.text('Output sequence length: ' +str(out_sqc))
                                st.text('Number of total features : ' +str(total_dim))
                                st.text('Number of temporal-covariates : ' +str(aux_dim))
                                st.text('Batch sz train/valid : ' +str(batch_tr))
                                st.text('Batch sz test : ' +str(batch_ts))

                                st.markdown("""---""")

                                # transform the data to tensors
                                tensor_train = data_loader(train, inp_sqc, out_sqc, aux_dim, batch_tr)
                                tensor_valid = data_loader(val, inp_sqc, out_sqc, aux_dim, batch_vl)                               
                                tensor_test = data_loader(test, inp_sqc, out_sqc, aux_dim, batch_ts)

                                # st.write(list(tensor_test)[0][-1])


                                # save tensors

                                save = st.text('preparing tensors for training...')

                                if not os.path.isdir('./train_tensor'):

                                        tf.data.experimental.save(tensor_train, './train_tensor', compression = 'GZIP')
                                        tf.data.experimental.save(tensor_valid, './valid_tensor', compression = 'GZIP')
                                        tf.data.experimental.save(tensor_test, './test_tensor', compression = 'GZIP')

                                        with open('meta_data.pkl','wb') as f:
                                                pickle.dump([train, scaler, lst_streets, streets, timestamp_test], f)

                                        save.text('saving tensors ... done.')

                                else:
                                        save.text('tensors already saved.')


                                if st.button("BACK"):
                                        st.text("Restarting...")

                                logging.info("-- prepare pipeline for tf")


                elif st.button('Model Training') or session_state.check1:
                        
                        st.subheader('Model Configuration')

                        hidd_dim, nb_epchs, rcr, krnl, dr, patience, delta = slider_display()
                        
                        save_updates_yaml(hidd_dim, nb_epchs, rcr, krnl, dr, patience, delta)

                        lstm_ed = LSTM_ED(total_dim, hidd_dim, rcr, krnl, dr)
                       
                        session_state.check1 = True

                        st.markdown("""---""")

                        if st.button("CONTINUE - TRAIN"):
                                st.subheader('Training')


                                tensor_train = tf.data.experimental.load('./train_tensor', compression='GZIP')
                                tensor_valid = tf.data.experimental.load('./valid_tensor', compression='GZIP')


                                with open('meta_data.pkl','rb') as f:
                                        train, scaler, lst_streets, streets, timestamp_test = pickle.load(f)

                                # start training
                                step_epoch = len(train) // batch_tr
                                
                                lstm_ed = training(lstm_ed, nb_epchs, step_epoch, # model, number of epochs, steps per epoch
                                        tensor_train,  tensor_valid, # training and validation tensors
                                        loss_fct, valid_loss_fct, opt, # loss functions and optimizer
                                        patience, delta) # early stopping

                                st.text("Save trained model...")
                                session_state.check1 = False

                                if st.button("BACK"):
                                        st.text("Restarting...")



                elif st.button('Model Inference'):

                        hidd_dim, nb_epchs, rcr, krnl, dr, patience, delta= upload_yaml()
                        lstm_ed = LSTM_ED(total_dim, hidd_dim, rcr, krnl, dr)
                        
                        # load model
                        checkpoint = tf.train.Checkpoint(model = lstm_ed)
                        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
                        
                        # # load tensor test and other info
                        tensor_test = tf.data.experimental.load('./test_tensor',  compression='GZIP')

                        #load datasets naive
                        with open('naive_datasets.pkl','rb') as f:
                                X_tr, Y_tr, X_vl, Y_vl, X_ts, Y_ts = pickle.load(f)

                                 
                        
                        #load meta data
                        with open('meta_data.pkl','rb') as f:
                                        train, scaler, lst_streets, streets, timestamp_test = pickle.load(f)

                        st.subheader('Testing')
                        

                        pred, targ, rmse, mae = testing(lstm_ed, tensor_test, aux_dim, scaler, out_sqc, lst_streets, streets, timestamp_test, X_tr, Y_tr, X_vl, Y_vl, X_ts, Y_ts)#[inp_sqc:]) 

        # logging.info("Finally, I can eat my pizza(s)")


if __name__ == "__main__":
        main()