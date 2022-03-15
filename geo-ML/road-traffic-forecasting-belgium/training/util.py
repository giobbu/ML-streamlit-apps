import tensorflow as tf
from util import logging

opt = tf.keras.optimizers.Adam()
loss_fct = tf.keras.losses.MeanAbsoluteError()
valid_loss_fct = tf.keras.losses.MeanAbsoluteError()


# early stopping

def monitor_loss(epoch, val_loss, min_delta, patience_cnt):
    if ((epoch > 0) and (val_loss[epoch-1] - val_loss[epoch])) > min_delta: 
        patience_cnt = 0 
    else: 
        patience_cnt += 1 
    return patience_cnt
 
