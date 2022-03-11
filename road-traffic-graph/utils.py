import re
from tqdm import tqdm
import pandas as pd
import holoviews as hv
import hvplot.networkx as hvnx


def clean_name_street(row):

    if type(row) == str:
        
        if row.find("-") != -1 :
            
            name = row.strip().split('- ')[0]
            
        else:
            
            name = row

    elif type(row) == list:
        
        if row[0].find("-") != -1 :
            
            name = row[0].strip().split('- ')[0]
            
        else:
            
            name = row[0]

    else:
        
        name = 'no_name'

    return name


def consecutive_streets( fix, roll ):

    
    # 1: beginning street
    
    # 2: end street
    
    if (fix[1] == roll[2]) and (fix[2] == roll[1]):

            u_1 = str(roll[0])
            v_1 = str(fix[0])
                        
            u_2 = str(fix[0])
            v_2 = str(roll[0])
            


    elif (fix[1] == roll[2]) and (fix[2] != roll[1]): 

            u_1 = str(roll[0])
            v_1 = str(fix[0])
            
            u_2 = 'no consecutive'
            v_2 = 'no consecutive'
            
                        
            
    elif (fix[1] != roll[2]) and (fix[2] == roll[1]): 

            u_1 = str(fix[0])
            v_1 = str(roll[0])
            
            u_2 = 'no consecutive'
            v_2 = 'no consecutive'
            
                               
    else:

            u_1 = 'no consecutive'
            v_1 = 'no consecutive'
            
            u_2 = 'no consecutive'
            v_2 = 'no consecutive'
            
            
    return  u_1, v_1, u_2, v_2 


def build_df_graph(DF):
    
    u_list = [ ]
    v_list = [ ]
    
#     weight_list = [ ]

    df = DF[[ 'node', 'first', 'last']].values

    for i in tqdm(range(len( df))):

        df_iter = df[ i + 1: , : ]

        fixed_row = df[ i , : ]

        for k in range(len( df_iter)):

            roll_row = df_iter[ k ]

            u_1, v_1, u_2, v_2 = consecutive_streets( fixed_row, roll_row )

            u_list.extend([ u_1, u_2 ]) #+= [ u_1, u_2 ] 
            v_list.extend([ v_1, v_2 ]) #+= [ v_1, v_2 ]
            
    df_u = pd.DataFrame(u_list, columns = ['u'])
    df_v = pd.DataFrame(v_list, columns = ['v'])
    
    # df_weight = pd.DataFrame (weight_list,columns=['weight'])
    # df_union = pd.concat([df_u ,df_v, df_weight], axis=1)

    dfg = pd.concat([ df_u, df_v ], axis=1)
    
    dfg = dfg[dfg.u != 'no consecutive']
    
    return dfg


def visualize_attr_graph(H_undir, pos, attr, nodes_size, w, h ):
    
    viz = hvnx.draw(H_undir, pos = pos, font_size = nodes_size, node_color = attr, cmap = 'Category10', width = w, height = h) 

    return viz 