import networkx as nx
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
from IPython.display import display
import holoviews as hv
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_folium import folium_static
from utils import build_df_graph, clean_name_street, visualize_attr_graph



class GraphOsm:

    """ 

    Retrieve road network from OpenStreetMap (OSM) and save it in different formats for later visualization.

    """
    
    def __init__(self, area, drive, query, buffer):
    # parameters to query OSM
        self.area = area     # area of interest (e.g. Namur, Belgium) 
        self.drive = drive   # typology of roads (e.g. drive)
        self.query = query   # roads of interest (e.g. motorways)
        self.buffer = buffer   # width of roads (e.g. 4.5 meters)


    def retrieve(self):
    # retrieve the streets with OSMnx module: 'ox.graph_from_place()'
        if  self.drive == 'none':
            if self.area == 'Namur, Belgium' or self.area == 'Mechelen, Belgium':
                # querying Namur/Mechelen returns two results: point geometry and (multi)polygon
                G = ox.graph_from_place(self.area, network_type = self.drive, custom_filter = self.query, which_result = 2)
            else:
                G = ox.graph_from_place(self.area, network_type = self.drive, custom_filter = self.query) 
        else:
            G = ox.graph_from_place(self.area, network_type = self.drive)
        G = ox.project_graph(G, to_crs = 'epsg:4326') # set coordinate reference system
        return G


    def deconstruct(self, G):
        intersections, streets = ox.graph_to_gdfs(G) 
        print('Graph representation of road network:')
        print('')
        print('streets: edges '+str(streets.shape))
        print('')
        print('street intersections: nodes '+str(intersections.shape))
        print('')
        return intersections, streets


    def vis_save_plot(self, G, name, node_size, edge_width):
    # plot and save in png
        fig, ax = ox.plot_graph(G, bgcolor='k', node_size = node_size, 
                                node_color='#999999', node_edgecolor='none', node_zorder=1,
                                edge_color='#555555', edge_linewidth = edge_width, edge_alpha=1)
        fig.savefig( name + '.png')
        return fig


    def vis_save_folium(self, G, name, edge_width ):
    # plot in folium and save in png
        m1 = ox.plot_graph_folium(G, popup_attribute= None, edge_color='blue', 
                                  edge_width= edge_width, edge_opacity=50)
        filepath = name + '_vis_folium.html'
        m1.save(filepath)
        return  folium_static(m1)


    def save_geojson(self, G, name):
    # save to GeoJSON
        edges = ox.graph_to_gdfs(G, nodes = False, edges = True)
        # set new coordinate reference system in meters
        ECKERT_IV_PROJ4_STRING = "+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        edges_meter = edges.to_crs(ECKERT_IV_PROJ4_STRING)
        # buffer the geometry of the streets
        edges_meter = edges_meter.geometry.buffer(self.buffer) 
        # reset the coordinate reference system back to the original
        edges_meter = edges_meter.to_crs({'init' :'epsg:4326'})
        streets_net = gpd.GeoDataFrame(edges_meter.geometry)
        streets_net.columns = ['geometry']
        streets_net.crs = {'init' :'epsg:4326'}
        streets_net.to_file(name + '.json', driver = 'GeoJSON')
    

    def save_pickle(self, G, intersections, streets, name):       
        with open( str(name) + '.pkl', 'wb') as f:
            pickle.dump([G, intersections, streets], f)



class GraphGCN:
    
    def __init__( self, streets ):
        self.streets = streets
                
        
    def prepare_streets_DF( self ):
        
        # first and last gps point in street segment 
        first_points = [Point(self.streets.iloc[ row ].geometry.coords[0]) 
                        for row in range(len( self.streets))]
        end_points = [Point(self.streets.iloc[ row ].geometry.coords[-1]) 
                       for row in range(len( self.streets))]
        
        self.streets['first'] = [ Point( x.coords[0][1], x.coords[0][0] ) for x in first_points ]
        self.streets['last'] = [ Point( x.coords[0][1], x.coords[0][0] ) for x in end_points ]
        
        # midpoint in street segment
        list_streets = [list(x.coords) for x in list(self.streets.geometry)]
        list_midpoint = [list_streets[i][int(len(list_streets[i])/2)] 
                          for i in range(len(list_streets))]
        
        self.streets['midpoint'] = [ ( x[1], x[0] ) for x in list_midpoint]
        
        # create a list of clean street names
        list_ = [clean_name_street(row) for row in self.streets['name']]
        
        # count consecutive street segments with same name
        self.streets['street_name'] = [row + '_' + str(list_[:index].count(row))  
                                       if row in list_[:index] else row 
                                       for index, row in enumerate(list_)]

        # select df columns
        streets_df = self.streets.reset_index()
        streets_df['node'] = streets_df['index'].astype( str ) + '_' + streets_df[ 'u' ].astype( str )  + '_' + streets_df[ 'v' ].astype( str )
        streets_df = streets_df[[ 'node', 'first', 'last', 'oneway', 'highway', 'street_name', 'midpoint', 'geometry', 'length' ]]
        return streets_df
    
  
    def build_graph_DF(self, streets_df, with_type, with_attr):
        df_streets = self.prepare_streets_DF()
        dfg = build_df_graph(df_streets)
        df_graph = dfg[[ 'u', 'v' ]]
        G = nx.Graph()

        if with_type == 'direct':
            G = nx.from_pandas_edgelist( df_graph, 'u', 'v', create_using = nx.DiGraph() )
            components = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)] 
            G_ = components[0]
            
        else:
            G = nx.from_pandas_edgelist(df_graph, 'u', 'v')
            components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
            G_ = components[0]

        streets_df = streets_df[streets_df['node'].isin(list(G_.nodes))] 
        streets_graph = list(streets_df.geometry)

        if with_attr == False:
            return streets_graph, df_graph, G_

        else:
            position_attr = {}
            for node in streets_df.values:
                position_attr[node[0]] = node[-3][::-1]
            nx.set_node_attributes(G_, position_attr, 'pos')

            # --------------------------------------

            type_street_attr = {}
            for node in streets_df.values: # Loop through the list, one row at a time
                if type(node[-5]) == list:
                    type_street_attr[node[0]] = node[-5][1]
                else:
                    type_street_attr[node[0]] = node[-5]
            nx.set_node_attributes(G_, type_street_attr, 'type_street')


            # --------------------------------------

            one_way_attr = {}
            for node in streets_df.values: 
                one_way_attr[node[0]] = node[-6]
            nx.set_node_attributes(G_, one_way_attr, 'one_way')

            # --------------------------------------

            relabeling = {}
            for node in streets_df.values: 
                relabeling[node[0]] = node[-4]
            H = nx.relabel_nodes(G_, relabeling)
            return streets_graph, df_graph, H
        
        
    def extract_Adjancency(self, H_):
        A_ = nx.to_numpy_matrix(H_)
        return A_


    def save_Gr_Adj_toPickle(self, H_, A_, name):
        with open( str(name) + '.pkl', 'wb') as f:
            pickle.dump([H_, A_], f)

    
    def viz_save_toNetworkx(self, H, pos, un_direct):
        plt.figure(figsize=(15, 15))       
        graph = nx.draw( H, pos = pos, node_size = 50)
        plt.savefig(un_direct + '.png')
        return graph


    def save_toHTML(self, H, pos, un_direct):
        viz = visualize_attr_graph( H, pos, 'type_street', nodes_size = 2, w = 700, h = 900)
        renderer = hv.renderer('bokeh')
        name = un_direct
        renderer.save(viz, name)
        return viz