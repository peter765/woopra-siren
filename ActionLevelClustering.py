'''
ActionLevelClustering.py
This script is designed to call the Woopra Search API to pull information
about current online users and their characteristics.

Fields of interest are extracted and placed in a database accordingly.

The training algorithm will access the data from the database in order
to classify further API calls.

Database structure:
visitor(~d):[{ //index visitors anonymously(can still look up by time)
    info:{
        app_platform:
        app_version:
        operating_system:
        screen_resolution:
    }
    action(~m):[{ //index actions by timestamp
        name/~n:
        ~m:
        },...]
    }
    actions: number of actions

##todo: Apply Dataset Augmentation.
#dataset can be augmented with samples that contain error-related behavior.
#these error-related samples will be different enough from the normal Dataset
#so they can be differentiated via clustering

##todo: Apply PCA and Feature Reduction
#no system to test PCA yet, in progress

},...]
'''
from APIProcess import APIProcess
import FeatureExtract
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import random as rand

woopra_data = APIProcess()
woopra_data.writeToFile()


#preprocess the api data to be used in ML clustering
x, features = FeatureExtract.nextActionFeatures2D(woopra_data,8)
f_indices = list(features.values())
f_names = list(features.keys())

#learning parameters
clusters = 8

#potentially PCA, other features
reduced_data = x


#KMeans model
model = KMeans(n_clusters=clusters).fit(reduced_data)

#Gaussian Mixture model
#model = GaussianMixture(n_components=clusters).fit(x)

predictions = [] #predictions stored in this array. determines color in graph
i=0
for value in x:
    predictions.append(model.predict(value.reshape(1,-1))[0]) #predicts model for each point
    i += 1

x = np.concatenate((x, np.array(predictions)[:,None]), axis=1)


#saves data to external JSON for debug
woopra_data.writeToFile()

##places results in 3d graph
fig = plt.figure()


#generate color pallete for clustersplt.legend(handles=legend_handles)
cluster_color = [] #based on selected cluster
legend_handles = []
for i in range(clusters):
    color = (rand.uniform(0.2,0.9), rand.uniform(0.2,0.9), \
    rand.uniform(0.2,0.9))
    cluster_color.append(color)
    legend_handles.append(mpatches.Patch(color=color, \
    label='Cluster {}'.format(i)))

#creates plot legend
plt.legend(handles=legend_handles)

if(len(f_names) == 3): #3d graph if given 3 features
    ax = Axes3D(fig)

    for i in range(0, len(x)): #plot each point + it's index as text above
        ax.scatter(x[i,f_indices[0]],x[i,f_indices[1]],x[i,f_indices[2]],\
        c=cluster_color[int(x[i,-1])])
        #ax.scatter(reduced_data[i,0],reduced_data[i,1],reduced_data[i,2],\
        #c=cluster_color[predictions[i]])

    centroids = model.cluster_centers_

    ax.scatter(centroids[:,f_indices[0]],centroids[:,f_indices[1]],\
    centroids[:,f_indices[2]], marker='x', s=169, \
    linewidths=6,color='k', zorder=10)

    ax.set_xlabel(f_names[0])
    ax.set_ylabel(f_names[1])
    ax.set_zlabel(f_names[2])
elif(len(f_names) == 2):
    ax = fig.add_subplot(111)
    for i in range(0, len(x)): #plot each point + it's index as text above
        color = cluster_color[int(x[i,-1])]
        ax.scatter(x[i,0],x[i,1],c=color)

    centroids = model.cluster_centers_

    ax.scatter(centroids[:,0],centroids[:,1], marker='x', color='k')

    ax.set_xlabel(f_names[0])
    ax.set_ylabel(f_names[1])

plt.show()
