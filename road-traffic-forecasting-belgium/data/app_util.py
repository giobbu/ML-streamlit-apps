import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import streamlit as st
import altair as alt

def plot_distribution(df, c, w, h):
    df_histogram = pd.DataFrame()
    df_histogram['mean'] = df.mean(axis=0)[:-4]
    arr = alt.Chart(df_histogram).mark_bar().encode(alt.X("mean:Q", bin=alt.Bin(maxbins=100)), y='count()',color=alt.value(c)).properties(width=w, height=h).interactive()
    return st.altair_chart(arr)



def plot_gantt(train, val, test):
    tot = train.shape[0] +  val.shape[0]+ test.shape[0]
    percentages = [train.shape[0]*100/tot, val.shape[0]*100/tot, test.shape[0]*100/tot]
    labels = ['train','validation','test']

    source = pd.DataFrame([
    {"task": "0_train", "start": 0, "end": train.shape[0]*100/tot, 'color':'blue'}, 
    {"task": "1_valid", "start": train.shape[0]*100/tot, "end": train.shape[0]*100/tot + val.shape[0]*100/tot, 'color':'orange'},
    {"task": "2_test", "start": train.shape[0]*100/tot + val.shape[0]*100/tot, "end": train.shape[0]*100/tot + val.shape[0]*100/tot + test.shape[0]*100/tot, 'color':'green'}])

    gnett = alt.Chart(source).mark_bar().encode(
        x='start',
        x2='end',
        y='task',
        color=alt.Color('color:N', scale = None)
    ).properties(height = 400, width = 700)

    return st.altair_chart(gnett)


def plot_tensor(x, y, z, dx, dy, dz, c):

    fig = plt.figure()
    ax = Axes3D(fig)

    xx = [x, x, x+dx, x+dx, x]
    yy = [y, y+dy, y+dy, y, y]

    kwargs = {'alpha': 0.8, 'color': c}
    ax.plot3D(xx, yy, [z], **kwargs) #*5
    ax.plot3D(xx, yy, [z+dz], **kwargs) #*5
    ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
    ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
    ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
    ax.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)

    ax.set_xlabel('Sequence Sz', fontsize=20)
    ax.set_ylabel('Features Sz', fontsize=20)
    ax.set_zlabel('Batch Sz', fontsize=20)
    ax.grid(False)

    return st.pyplot(fig)
     