import numpy as np 


def inverse_transform(pred, scaler):
    # invert scaling
    inv_pred = scaler.inverse_transform(pred)
    return inv_pred

def rmse_metric(true, pred):
    rmse = np.sqrt((np.square(true - pred)).mean(axis=0))
    return rmse


def mae_metric(true, pred):
    mae = np.absolute(true - pred).mean(axis=0)
    return mae

def evaluation_fct(targets, forecasts, n_seq):
    
    list_rmse = []
    list_mae = []
    
    for i in range(n_seq):

        true = np.vstack([target[i] for target in targets])
        pred = np.vstack([forecast[i] for forecast in forecasts])
        
        rmse = rmse_metric(true, pred)
        mae = mae_metric(true, pred)
        
        list_rmse.append(rmse)
        list_mae.append(mae)
        
    list_rmse = np.vstack(list_rmse)
    list_mae = np.vstack(list_mae)
    
    return list_rmse, list_mae



