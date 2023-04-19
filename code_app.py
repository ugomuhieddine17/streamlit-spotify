#####################################################
################## PACKAGES #########################
#####################################################
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import networkx as nx
from pyvis.network import Network
import sys 
import base64
from music_utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as ntx
import csv
import random
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta, date
import pandas as pd
from itertools import combinations
from numpy.linalg import norm
import pickle
from music_utils import *
import os
from torch_geometric.data import Data
import torch
import torch_geometric.nn as graphnn
import torch.nn as nn
import torch
from torch.nn import Linear
import torch.nn.functional as F


local = True
colab = False
git = False
if local:
    DATA_PATH = './data/'

elif git:
    DATA_PATH = './data/'
    
elif colab:
    from google.colab import drive
    import sys
    DATA_PATH = './gdrive/MyDrive/MLNS_Spotify/data/'
    drive.mount('/content/gdrive', force_remount=True)
    sys.path.append('/content/gdrive/MyDrive/MLNS_Spotify')

PATH_TRAIN = DATA_PATH+"train.txt"
PATH_NODE_INFO = DATA_PATH+"node_information.csv"
PATH_TEST = DATA_PATH+"test.txt"
# sys.path.append('/content/gdrive/MyDrive/MLNS_Spotify')



class MLP_post(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        torch.manual_seed(12345)
        self.num_layers = num_layers
        self.lin1 = Linear(in_channels, hidden_channels)
        self.list_FC = nn.ModuleList()

        for i in range(num_layers):
            self.list_FC.append(nn.Linear(hidden_channels, hidden_channels))
        
        self.last_lin = Linear(hidden_channels, out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        for i in range(self.num_layers):
            x = F.elu(self.list_FC[i](x))
            x = F.dropout(x, p=0.5, training=self.training)
        return self.softmax(self.last_lin(x))
        

class GAT_MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, embed_size, n_heads,  MLP_num_layers, MLP_hidden_channels, MLP_out_channels, dropout=False):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout

        self.num_layers = num_layers
        self.graphconv1 = graphnn.conv.GATConv(in_channels=in_channels, out_channels=hidden_channels, heads=n_heads, concat=True)
        self.list_GATC = nn.ModuleList()

        for i in range(num_layers):
            self.list_GATC.append(graphnn.conv.GATConv(in_channels=n_heads*hidden_channels, out_channels=hidden_channels, heads=n_heads, concat=True))
        
        self.last_conv = graphnn.conv.GATConv(in_channels=n_heads*hidden_channels, out_channels=embed_size, heads=n_heads, concat=False)

        self.elu = nn.LeakyReLU(negative_slope=0.2)
        self.MLP_post = MLP_post(in_channels=embed_size*2, hidden_channels=MLP_hidden_channels, num_layers=MLP_num_layers, out_channels=MLP_out_channels)
                  
    def forward(self, x, edge_index):
          # print('very first', x)
        # TO AVOID NAN PROPAGATION
          x = torch.nan_to_num(x)
          # print('first', x)
          x = self.graphconv1(x, edge_index)
        #   print('x beginnning', x)
          x = self.elu(x)
          for i in range(self.num_layers):
              x = x + self.elu(self.list_GATC[i](x, edge_index))
              x = F.dropout(x, p=0.2, training=self.training)

          x = self.last_conv(x, edge_index).relu()
          # print('end GAT', x)
          #we concatenate
          # Extract the source and target node indices from edge_index
          src_idx = edge_index[0]
          tgt_idx = edge_index[1]

          # Use indexing to extract the node features for the source and target nodes
          src_features = x[src_idx]
          tgt_features = x[tgt_idx]

          # Concatenate the features along the last dimension
          x = torch.cat([src_features, tgt_features], dim=-1)
          # print('concatenated', x.size())
          x = self.MLP_post.forward(x)
          # print('final', x)
        #   print('x',x)
          return x

@st.cache_data
def full_initialisation():
    ######################################################
    ######################################################
    ##               DATA INITIALISATION               ###
    ######################################################
    ######################################################


    spotify_600, artists_600 = read_spotify_600(DATA_PATH=DATA_PATH, read=True)


    #####
    ## PARAMETERS
    #####
    start_date = datetime.strptime("1998-01-01 00:00:01", "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime("2020-12-31 23:59:00", "%Y-%m-%d %H:%M:%S")
    n_month = 12
    interval = timedelta(days=365.25*n_month/12)

    # one to one spotify database rearrangement
    start_date_spotify_600 = spotify_600[spotify_600.release_date.dt.year >= start_date.year].copy()
    spot_600 = start_date_spotify_600.copy()

    song_artist_pairs = [(track_id, artist_pair[0], artist_pair[1]) for track_id, artists in spot_600[['track_id', 'id_artists']].values for artist_pair in combinations(set(artists), 2)]
    correspondace_spot_600 = pd.DataFrame(song_artist_pairs, columns=['track_id', 'artist_1', 'artist_2'])
    spot_600 = pd.merge(correspondace_spot_600, spot_600.drop(columns=['name', 'artists', 'id_artists', 'artist_id']), on='track_id', how='left').copy()

    # existing artists 
    in_spot_artists_600 = artists_600[artists_600.artist_id.isin(start_date_spotify_600.id_artists.explode().unique())].copy()


    ######################################################
    ## BUILD FEATURES OF ARTIST CONSIDERING START DATE ###


    #features of the artists
    artist_features = artists_features_creation(in_spot_artists_600,
                                            start_date_spotify_600,
                                            start_date_spotify_600,
                                            DATA_PATH, read=False,
                                            pkl_features_artist_path='features_artists_PYGT_yt_1999.pkl',
                                ).reset_index()

    # print(f'len of artist_features : {len(artist_features)}')

    ######################################################
    ##          ARTIST ID TO INT DICTIONNARY           ###

    # Reassign the location IDs (makes it easier later, because here the IDs didn't start at 0)
    artist_idname = artist_features['artist_id'].unique()
    new_ids = list(range(len(artist_idname)))
    mapping = dict(zip(artist_idname, new_ids))
    reversed_mapping = dict(zip(new_ids, artist_idname))

    artist_features['int_artist_id'] = artist_features['artist_id'].map(mapping)


    spot_600['artist_1'] = spot_600['artist_1'].map(mapping)
    spot_600['artist_2'] = spot_600['artist_2'].map(mapping)

    #We drop potential nans
    missings = spot_600[(spot_600.artist_1.isna()) | (spot_600.artist_2.isna())].copy()
    spot_600 = spot_600.dropna(subset=['artist_1', 'artist_2']).copy()


    int_to_name = dict(artist_features[['int_artist_id', 'name']].values)

    spot_600['artist_1_name'] = spot_600.artist_1.map(int_to_name)
    spot_600['artist_2_name'] = spot_600.artist_2.map(int_to_name)


    df_featurings = spot_600.groupby(['artist_1', 'artist_2']).agg(num_feats=('track_id', 'count')).reset_index()

    df_featurings['artist_1_name'] = df_featurings.artist_1.map(int_to_name)
    df_featurings['artist_2_name'] = df_featurings.artist_2.map(int_to_name)

    node_features = np.array(artist_features.drop(columns=['artist_id', 'genres', 'name', 'int_artist_id']).fillna(0))
    node_features = (node_features - node_features.mean(axis=0))/node_features.std(axis=0) 

    model = torch.load(DATA_PATH + 'best-model_very_good_92accu.pt',  map_location='cpu')

    return mapping, reversed_mapping, int_to_name, spot_600, artist_features, df_featurings, node_features, model, start_date_spotify_600, in_spot_artists_600


def artist_features_evolving(in_spot_artists_600, start_date_spotify_600, end_date, mapping):

    start_end_spot_600 = start_date_spotify_600[start_date_spotify_600.release_date <= end_date].copy()
    artist_features = artists_features_creation(in_spot_artists_600,
                                            start_date_spotify_600,
                                          start_end_spot_600,
                                          DATA_PATH, read=False
                              ).reset_index()
    artist_features['int_artist_id'] = artist_features['artist_id'].map(mapping)
    node_features = np.array(artist_features.drop(columns=['artist_id', 'genres', 'name', 'int_artist_id']).fillna(0))
    node_features = (node_features - node_features.mean(axis=0))/node_features.std(axis=0) 

    return node_features

def test_Data_construction(df_select, node_features):
    """build the data test obect

    Args:
        df_select (pandas dataframe): the selected spotify_600 subset
        node_features (array): features of the nodes

    Returns:
        pytorch geometric Data: the test Data object
    """
    
    edge_list = torch.from_numpy(np.array(df_select[['artist_1','artist_2']].values).transpose())
    edge_attr = torch.from_numpy(np.array(df_select.num_feats.values).transpose())
    y = torch.from_numpy(np.array(df_select.done_feat.values).transpose())
    test_data = Data(x=torch.from_numpy(node_features).float(), 
    y_indices=edge_list.long(), 
    edge_index=edge_list, 
    edge_attr=edge_attr,
    y=y)
    
    return test_data


def visualize_val_prediction(val_data, int_to_name, graph_name='val_graph'):

    got_net = Network(height='1000px', width='100%',bgcolor='#222222', font_color='white',
    notebook = False, directed=False)

    print('bite')
    sources = val_data.y_indices[0,:].tolist()
    targets = val_data.y_indices[1,:].tolist()
    prediction = val_data.prediction.tolist()
    true_label = val_data.y.tolist()
    
    sources = [int_to_name[x] for x in sources]
    targets = [int_to_name[x] for x in targets]

    # true_label = 
    edge_data = zip(sources, targets, prediction, true_label) 

    for e in edge_data:
        src = str(e[0])
        dst = str(e[1])
        pred = e[2]
        label = e[3]
        got_net.add_node(src, src, title=src)
        got_net.add_node(dst, dst, title=dst)
    
        if pred+label == 2:
            got_net.add_edge(src, dst, title="TP", color="#38761d", width=4)

        if pred+label == 0:
            got_net.add_edge(src, dst, title="TN", color="#b6d7a8", width=4)
            pass
        elif pred == 0 and label == 1:
            got_net.add_edge(src, dst, title="FN", color="#f44336", width=4)

        elif pred == 1 and label == 0:
            got_net.add_edge(src, dst, title="FP", color="#f4cccc", width=4)

    try:
        path = '/tmp'
        got_net.save_graph(f'{path}/{graph_name}.html')
        HtmlFile = open(f'{path}/{graph_name}.html', 'r', encoding='utf-8')
        print('tmp')

    # Save and read graph as HTML file (locally)
    except:
        path = './html_files'
        got_net.save_graph(f'{path}/{graph_name}.html')
        HtmlFile = open(f'{path}/{graph_name}.html', 'r', encoding='utf-8')

    # print(artist_net)
    # Load HTML file in HTML component for display on Streamlit page
    # print(HtmlFile)
    raw_html = HtmlFile.read().encode("utf-8")
    raw_html = base64.b64encode(raw_html).decode()
    components.iframe(f"data:text/html;base64,{raw_html}", height=510)#, width=700)


def y_labels_val(spot_600, df_select):
    labels_df = spot_600[(spot_600.release_date >= begin_date) & (spot_600.release_date <= end_date)].copy()
    st.markdown('labels_df')
    
    labels_df = labels_df.groupby(['artist_1_name', 'artist_2_name']).agg(num_feats=('track_id', 'count')).reset_index()
    labels_df['done_feat'] = labels_df.num_feats.apply(lambda x: 1 if x >= 1 else 0)
    st.markdown(sum(labels_df.done_feat))
    st.table(labels_df.head())
    df_select = pd.merge(df_select, labels_df[['artist_1_name', 'artist_2_name', 'done_feat']],
                        on=['artist_1_name', 'artist_2_name'],
                        how='left'
                    )
    df_select.done_feat = df_select.done_feat.fillna(0)

    st.markdown('merged selected')
    st.markdown(sum(df_select.done_feat))
    st.table(df_select.head())
    return df_select


mapping, reversed_mapping, int_to_name, spot_600, artist_features, df_featurings, node_features, model, start_date_spotify_600, in_spot_artists_600 = full_initialisation()


######################################################
######################################################
##          GRAPH USER PARAMETERIZATION            ###
######################################################
######################################################

# Set header title
st.title('Network Graph Visualization of artist interactions')

# Define list of selection options and sort alphabetically
artist_list = artist_features.name.unique()

#Define the list of genres
genres_list = artist_features.genres.explode().unique()

with st.sidebar:
    #define the genre list
    graph_type = st.radio(
        "What graph do you want to display?",
        ('Direct connections', 'Propagated connections (order 2)', 'Genres'))

    begin_date = st.date_input(
    "Begin date for validation",
    date(2019, 1, 1))
    
    end_date = st.date_input(
    "End date for validation",
    date(2020, 1, 1))

    begin_date = np.datetime64(begin_date)
    end_date = np.datetime64(end_date)

    node_features_val = artist_features_evolving(in_spot_artists_600, start_date_spotify_600, end_date, mapping)

    if graph_type != 'Genres':
        # Implement multiselect dropdown menu for option selection (returns a list)
        selected_artists = st.multiselect('Select artist(s) to visualize', artist_list)

        # Set info message on initial site load
        if len(selected_artists) == 0:
            st.text('Choose at least 1 artist to get started')

        else:
            if  graph_type == 'Direct connections':
                df_select = df_featurings.loc[df_featurings['artist_1_name'].isin(selected_artists) | \
                                            df_featurings['artist_2_name'].isin(selected_artists)]
                df_select = df_select.reset_index(drop=True)
                df_select = y_labels_val(spot_600, df_select)
                
            else:
                df_pre_select = df_featurings.loc[df_featurings['artist_1_name'].isin(selected_artists) | \
                                            df_featurings['artist_2_name'].isin(selected_artists)]
                df_pre_select = df_pre_select.reset_index(drop=True)

                propagated_list = list(set(list(df_pre_select.artist_1_name.unique()) + list(df_pre_select.artist_2_name.unique())))

                df_select = df_featurings.loc[df_featurings['artist_1_name'].isin(propagated_list) | \
                                            df_featurings['artist_2_name'].isin(propagated_list)]
                df_select = y_labels_val(spot_600, df_select)
                
    elif graph_type == 'Genres':
        selected_genres = st.multiselect('Select Genre(s) to visualize', genres_list)

        if len(selected_genres) == 0:
            st.text('Choose at least 1 genre to get started')

        else:
            df_pre_select = artist_features.explode('genres')
            selected_artists = df_pre_select[df_pre_select.genres.isin(selected_genres)].name.unique()
            df_select = df_featurings.loc[df_featurings['artist_1_name'].isin(selected_artists) | \
                                        df_featurings['artist_2_name'].isin(selected_artists)]
            df_select = df_select.reset_index(drop=True)
            
            df_select = y_labels_val(spot_600, df_select)


    st.table(df_select.head())

#plot the most probable featurings
# 

# choice = st.number_input("Pick the number of most probable featurings", 0, 50)
if st.button('Display the predictions'):
    test_data = test_Data_construction(df_select, node_features)
    test_pred = model(test_data.x, test_data.y_indices)
    proba_featuring = test_pred[:,1].tolist()
    proba_featuring = [round(prob,3) for prob in proba_featuring]
    sources = test_data.y_indices[0,:].tolist()
    targets = test_data.y_indices[1,:].tolist()

    sources = [int_to_name[x] for x in sources]
    targets = [int_to_name[x] for x in targets]

    edge_data = zip(sources, targets, proba_featuring) 

    ### THE LIST  ###
    #################
    # edge_data = sorted(edge_data, key=lambda x: x[2])
    edge_df = pd.DataFrame(edge_data, columns=['artist 1', 'artist 2', 'probability'])
    edge_df.probability = edge_df.probability.apply(lambda x: round(x*100,1))
    # if len(choice) == 0:
    #     st.table(edge_df.sort_values(by='probability')[:3])
    # else:
    

    #################
    ### THE GRAPH ###
    #################
    artist_net = Network(height='1000px', width='100%',bgcolor='#222222', font_color='white',
    notebook = False, directed=False)


    for src, dst, proba in zip(sources, targets, proba_featuring):
        # print('e', src)
        src = str(src)
        dst = str(dst)
        # src = str(e[0])
        # dst = str(e[1])
        # proba = e[2]

        artist_net.add_node(src, src, title=src, font_size=60)
        artist_net.add_node(dst, dst, title=dst, font='60px arial black')
        if proba < 0.7 and  proba >= 0.45:
            artist_net.add_edge(src, dst, title=str(round(proba*100, 3))+' %', color="#F7D060", width=round(proba**2*15)+1)
        elif proba < 0.45:
            artist_net.add_edge(src, dst, title=str(round(proba*100, 3))+' %', color="#FF6D60", width=round(proba**2*15)+1)

        elif proba >= 0.7:
            artist_net.add_edge(src, dst, title=str(round(proba*100, 3))+' %', color="#98D8AA", width=round(proba**2*15)+1)


    artist_net.repulsion(node_distance=420, central_gravity=0.33,
                    spring_length=110, spring_strength=0.10,
                    damping=0.95)

    try:
        path = '/tmp'
        artist_net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')
        print('tmp')

    # Save and read graph as HTML file (locally)
    except:
        path = './html_files'
        artist_net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

    # print(artist_net)
    # Load HTML file in HTML component for display on Streamlit page
    # print(HtmlFile)
    raw_html = HtmlFile.read().encode("utf-8")
    raw_html = base64.b64encode(raw_html).decode()
    components.iframe(f"data:text/html;base64,{raw_html}", height=510)#, width=700)

    st.table(edge_df.sort_values(by='probability', ascending=True)[:30])


if st.button('Display validation set'):
    val_data = test_Data_construction(df_select, node_features_val)
    # val_data.y_indices, val_data.y = neg_sampling(val_data)
    val_proba = model(val_data.x, val_data.y_indices)
    val_pred = torch.argmax(val_proba, dim=1)
    val_data['prediction'] = val_pred
    st.markdown(f'### The accuracy \n \
     the accuracy is {round(accuracy_score(val_data.prediction.numpy(), val_data.y.numpy())*100, 2)} %. \
        \n {val_data}')

    visualize_val_prediction(val_data, int_to_name, graph_name='val_graph')




if st.button('Display the original graph'):

    G = nx.from_pandas_edgelist(df_select, 'artist_1_name', 'artist_2_name', 'num_feats')
    # Initiate PyVis network object
    print(G)
    artist_net = Network(height='500px', width='100%', bgcolor='#222222', font_color='white')
    

    # Take Networkx graph and translate it to a PyVis graph format
    artist_net.from_nx(G)

    # Generate network with specific layout settings
    artist_net.repulsion(node_distance=420, central_gravity=0.33,
                        spring_length=110, spring_strength=0.10,
                        damping=0.95)
    # Save and read graph as HTML file (on Streamlit Sharing)
    # try:
    #     path = '/tmp'
    #     artist_net.save_graph(f'{path}/pyvis_graph.html')
    #     HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')


    # # Save and read graph as HTML file (locally)
    # except:
    path = './html_files'
    artist_net.save_graph(f'{path}/pyvis_graph_2.html')
    HtmlFile = open(f'{path}/pyvis_graph_2.html', 'r', encoding='utf-8')

    # Load HTML file in HTML component for display on Streamlit page
    print(HtmlFile)
    raw_html = HtmlFile.read().encode("utf-8")
    raw_html = base64.b64encode(raw_html).decode()
    components.iframe(f"data:text/html;base64,{raw_html}", height=510)#, width=700)
