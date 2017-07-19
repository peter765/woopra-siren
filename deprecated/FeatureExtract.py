'''
FeatureExtract.py

Contains functions for accepting APIProcess objects.

Returns the following :
    -x: processed dataset(n*m numpy matrix, n samples, m columns)
    -key_features : 1x2 or 1x3 dict, indexes and axis labels for plotting

##todo: Apply Dataset Augmentation.
#dataset can be augmented with samples that contain error-related behavior.
#these error-related samples will be different enough from the normal Dataset
#so they can be differentiated via clustering

##todo: Apply PCA and Feature Reduction
#no system to test PCA yet, in progress

},...]
'''
from APIProcess import APIProcess
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import random as rand

'''
nextActionFeatures2D(api_data, act_clusters)
determines type of visit by finding
-current action
-immediately proceeding action

takes in APIProcess object
returns the following:
    x: (n,m) nparray, n samples with m features
    key_features:(dict with 3 val) numerical col indices in x and axis labels
    for graphed features
'''

def nextActionFeatures2D(api_data, act_clusters):
    #append timestamp(ms), elapsed_time, act#, action, n_axt#, next_action
    action_dict = api_data.actions
    visitors_data = api_data.visitors

    ##activity clustering
    #learning parameters

    #KMeans model
    action_model = KMeans(n_clusters=act_clusters)

    #list to hold features for all actions conducted in dataset
    all_features = []
    #features: event, next_event

    #extract sequence data from api_data and place in all_features
    for visitor in visitors_data:
        visitor_actions = visitor['events'] #dictionary that stores visitor data
        for item in visitor_actions['list']:
            features = []
            features.append(item[2]) #current action
            features.append(item[4]) #next action
            all_features.append(features)

    x = np.array(all_features)

    #train model to cluster actions
    predictions = action_model.fit_predict(x)

    x = np.hstack((x, predictions[:,None]))
    np.savetxt("action_clusters.csv",x, fmt='%i')

    ##graph action clusters
    ##copy code from earlier, refactor later
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #generate color pallete for clustersplt.legend(handles=legend_handles)
    cluster_color = [] #based on selected cluster
    legend_handles = []
    for i in range(act_clusters):
        color = (rand.uniform(0.2,0.9), rand.uniform(0.2,0.9), \
        rand.uniform(0.2,0.9))
        cluster_color.append(color)
        legend_handles.append(mpatches.Patch(color=color, \
        label='Cluster {}'.format(i)))
    #creates plot legend
    plt.legend(handles=legend_handles)

    for i in range(0, len(x)): #plot each point + it's index as text above
        color = cluster_color[int(x[i,-1])]
        ax.scatter(x[i,0],x[i,1],c=color)
        #ax.scatter(reduced_data[i,0],reduced_data[i,1],reduced_data[i,2],\
        #c=cluster_color[predictions[i]])


    centroids = action_model.cluster_centers_

    ax.scatter(centroids[:,0],centroids[:,1], marker='x', color='k')

    ax.legend(handles=legend_handles)

    ax.set_xlabel('current')
    ax.set_ylabel('next')

    plt.show()

    x = [] #placeholder vector
    #now label each visit
    for value in visitors_data:
        actions = []
        features = np.zeros((1, 3)) #placeholder vector
        base_time = value['time']
        total_time = 0

        for user_action in value['events']['list']:
            index = np.array([user_action[2], user_action[4]]).reshape(1,-1)
            label = int(action_model.predict(index))
            actions.append(label)
            total_time += (user_action[0]- base_time)

        features[0][0] = Counter(actions).most_common(1)[0][0] #most common action class
        features[0][1] = int(actions[0])#first action
        features[0][2] = value['events']['count'] #number of events
        x.append(features)

    x = np.squeeze(x)
    preprocessing.scale(x[:, 1], copy=False) #scales time column with zero mean

    key_features = {'visit label' : 0 , 'first action':1, '#actions' : 2}

    return x, key_features
