import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import numpy as np





def naive(history, seasonality, n_seq):

    list_seas = []
    
    # seas_0 = 168*2*2*2+23
    # seas_1 = 168*2*2+168*2+23
    seas_2 = 168*2*2 
    seas_3 = 168*2 
    

    
    list_seas_0 = []
    list_seas_1 = []
    list_seas_2 = []
    list_seas_3 = []
    
    # if seasonality > history.shape[0]:
        
    #     for i in reversed(range(1,24*2+23 +1)):
    #         season = history[-i][-1]
    #         list_seas.append(season)
    #         array_pred = np.vstack(list_seas)
    # else:
        
    # for i in reversed(range(seas_1+1,seas_0+1)):
    #     season_0 = history[-i][-1]
    #     list_seas_0.append(season_0)
        
    # for i in reversed(range(seas_2+1,seas_1+1)):
    #     season_1 = history[-i][-1]
    #     list_seas_1.append(season_1)
        
    for i in reversed(range(seas_3+1,seas_2+1)):
        season_2 = history[-i][-1]
        list_seas_2.append(season_2)
        
    for i in reversed(range(1,seas_3+1)):
        season_3 = history[-i][-1]
        list_seas_3.append(season_3)
        
#  np.vstack(list_seas_0),
#  np.vstack(list_seas_1),
    array_pred = np.vstack(list_seas_3)
            
    return array_pred[:n_seq]








class LSTM_ED(tf.keras.Model):

    def __init__(self, tot_dim, hidd_dim, rcr_init, reg,
                       drop_rt):

        super(LSTM_ED, self).__init__()

        self.tot_dim = tot_dim
        self.hidd_dim = hidd_dim
        self.rcr_init = rcr_init
        self.reg = reg
        self.drop_rt = drop_rt

        
        self.lstmE = tf.keras.layers.LSTM(self.hidd_dim,
                                         return_sequences=False,
                                         return_state = True,
                                         recurrent_initializer= self.rcr_init,
                                         kernel_regularizer = regularizers.l2(self.reg),
                                         activation = 'relu',
                                         name='Encoder')

        self.lstmD = tf.keras.layers.LSTM(self.hidd_dim, 
                                          return_sequences = True,
                                          recurrent_initializer= self.rcr_init,
                                          kernel_regularizer=regularizers.l2(self.reg),
                                          activation = 'relu',
                                          name ='Decoder')



        self.drop = tf.keras.layers.Dropout(self.drop_rt)

        self.dense0 = tf.keras.layers.Dense(self.hidd_dim,
                                            activation = 'relu', 
                                            kernel_regularizer=regularizers.l2(self.reg))
          
        self.dense = tf.keras.layers.Dense(self.tot_dim, 
                                           kernel_regularizer=regularizers.l2(self.reg))
                                           

    def __call__(self, inp_e, inp_d,  training=False):

        inp_e = tf.cast(inp_e, tf.float32)
        inp_d = tf.cast(inp_d, tf.float32)


        # encoder 1
        _, h, c = self.lstmE(inp_e)

        # decoder0
        out_d = self.lstmD(inp_d, initial_state= [h, c])


        if training:
            out_d = self.drop(out_d, training=training)

        # dense
        out_d = self.dense0(out_d)

        if training:
            out_d = self.drop(out_d, training=training)

        # dense
        out = self.dense(out_d)

        return out

