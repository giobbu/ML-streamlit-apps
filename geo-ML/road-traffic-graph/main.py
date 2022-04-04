import os
import sys
import argparse
from turtle import width
from _OSMNX_GCN_ import G_fromOSM, G_forGCN
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import streamlit.components.v1 as components

st.set_option('deprecation.showPyplotGlobalUse', False)


PATH_HTML = '/Html/'
PATH_JSON = '/Json/'
PATH_PCKL = '/Pickle/'
PATH_PNG = '/Png/'




def _setup_parser():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-a', type=str, default = 'Brussels Capital Region, Belgium', help='Area of interest (default is  "Brussels Capital Region, Belgium").')
    
    parser.add_argument('-d', type=str, default = 'none', help='Options: "drive" or "none". If "none" is selected then insert argument "-r".')

    parser.add_argument('-r', type=str, default = '["highway"~"motorway|primary"]', help = 'Insert Overpass query (default is  "["highway"~"motorway"]" ). Further information on https://stackoverflow.com/questions/61881345/how-to-import-multiple-infrastructure-type-in-osmnx/61897000#61897000')
    
    parser.add_argument('-b',  type=float, default = 4.5, help = 'Road width. Default vale is 4.5 meters')

    parser.add_argument('-type',  type=str, default = 'undirect', help = ' Type of graph to build. Options: "direct" or "undirect".')

    return parser



def main():
    
    # Parse arguments
    parser = _setup_parser()

    args = parser.parse_args()

    try:

        args = parser.parse_args()

    except SystemExit as e:

        os._exit(e.code)

    st.title("Road Network representation for GCN Learning")

    # Assign arguments to variables
    area = args.a

    drive = args.d

    road = args.r

    buffer = args.b

    un_direct = args.type


    area = st.selectbox(
    'SELECT AREA OF INTEREST', ('Namur, Belgium', 'Brussels Capital Region, Belgium', 'Mechelen, Belgium'))

    area_string = area.replace(',', '').split()[0]

    PATH_CITY = './' + area_string + '/'

    un_direct = st.selectbox(
    'CHOOSE GRAPH TYPE', ('undirect', 'direct'))
    
    retrieve = st.button('RETRIEVE STREETS')

    if 'count' not in st.session_state:
        
        st.session_state.count = 0

    if retrieve:
        
        st.header("Retrieve streets from OSM")

        code = '''# 
        g = G_fromOSM(area, drive, road, buffer)
        G = g.retrieve()
        intersections, streets = g.deconstruct(G)'''
        
        st.code(code, language='python')

        st.session_state.count = 1
        
        g = G_fromOSM(area, drive, road, buffer)

        G = g.retrieve()

        print(G)

        intersections, streets = g.deconstruct(G)

        st.session_state.streets = streets

#----------------------------------------------------------

        st.header("Visualize street network on OSMNX")

        code = '''# 
        fig = g.viz_save_toJPEG(G, [ AREA SELECTED ], node_size = 10,  edge_width = 0.5)
        st.pyplot(fig) '''
        
        st.code(code, language='python')
        
        # make html directory
        PATH_ = PATH_CITY + PATH_PNG
        isExist = os.path.exists(PATH_)

        if not isExist:
            os.makedirs(PATH_)

        fig = g.viz_save_toPNG(G, PATH_ + 'streets_' + area_string, node_size = 10,  edge_width = 0.5)
            
        st.pyplot(fig)

#----------------------------------------------------------

        st.header("Visualize street network on Folium")

        code = '''# 
        g.viz_save_toFolium(G, PATH_ + area_string, edge_width = 0.5)
        '''

        st.code(code, language='python')

        # make html directory
        PATH_ = PATH_CITY + PATH_HTML
        isExist = os.path.exists(PATH_)

        if not isExist:
            os.makedirs(PATH_)

        m = g.viz_save_toFolium(G, PATH_ + area_string, edge_width = 0.5)

        

#----------------------------------------------------------

        # make json directory
        PATH_ = PATH_CITY + PATH_JSON
        isExist = os.path.exists(PATH_)

        if not isExist:
            os.makedirs(PATH_)

        # save to json format
        g.save_toGeoJSON(G, PATH_ + area_string)


        # make pickle directory
        PATH_ = PATH_CITY + PATH_PCKL
        isExist = os.path.exists(PATH_)

        if not isExist:
            os.makedirs(PATH_)

        # save to pickle format
        g.save_toPickle(G, intersections, streets, PATH_ + area_string)


# ------------------------------------------------------------


    constr = st.button('CONSTRUCT GRAPH')

    if constr:
        
        if st.session_state.count == 0:

            st.write('RETRIEVE the streets from OSM first !')

        else:

            st.header('Employ streets retrieved from OSMNX')

            streets = st.session_state.streets
        
            g = G_forGCN(streets)

            df_streets = g.prepare_streets_DF()

            streets_graph, df_graph, H = g.build_graph_DF(df_streets, with_type = un_direct, with_attr = True)

            code = '''# 
            g = G_forGCN(streets)
            df_streets = g.prepare_streets_DF()
            streets_graph, df_graph, H = g.build_graph_DF(df_streets, with_type = un_direct, with_attr = True)'''
            
            st.code(code, language='python')

# ---------------------------------------------------------

            pos = nx.get_node_attributes( H, 'pos')

#----------------------------------------------------------

            PATH_ = PATH_CITY + PATH_PNG

            st.header('Visualize graph in Networkx')

            fig = g.viz_save_toNetworkx(H, pos, PATH_  + 'graph_' +  un_direct)

            st.pyplot(fig)

#----------------------------------------------------------

            st.header('Visualize graph in Holoviews')

            # save html to directory
            PATH_ = PATH_CITY + PATH_HTML
            
            viz = g.save_toHTML(H, pos, PATH_ + 'graph_' + un_direct)

            code = '''# 
            pos = nx.get_node_attributes( H, 'pos')
            g.save_toHTML(H, pos, un_direct)
            '''

            st.code(code, language='python')

#----------------------------------------------------------

            HtmlFile = open(PATH_ + 'graph_' + un_direct + '.html', 'r', encoding='utf-8')
            
            source_code = HtmlFile.read() 
            
            components.html(source_code, height = 900,  width = 900)



if __name__ == '__main__':
    
    main()
