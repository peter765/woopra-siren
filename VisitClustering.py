'''
VisitClustering.py
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

NOTES:
-ignores actions on iOS/Windows phone due to duplicate actions
-ignores property-update(potential noise)
},...]
'''
import json
import requests
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import random as rand


#json parameters for HTTP POST request
payload = {'website' : 'mycricket.at.cricketwireless.com'}
api_url = "http://www.woopra.com/rest/2.4/online/list"
#login info for woopra
app_ID = "FQ4MZO2VBCOIGIDTX44AXTPYHRFBYAY2"
secret = "y0QENfwHXXYea7ZvvEbgImeLncWXxzG4DvPDvpMrCkzwu3JjnzHCuVNwvHAjtXaq"
api_auth = (app_ID, secret)

#request data from server
req = requests.post(api_url, auth=api_auth, params=payload)
#encodes results as json
try:
    resp = req.json()
except JSONDecodeError:
    print("Unable to connect to network")
    exit()


#action map: maps every action type from a string to an int
action_dict = {}

#visitors data: for each visitor, record device info and an action list
#keys are ????, values are formatted as dictionary
visitors_data = []

i = 0
for visitor in resp['visitors']:
    if visitor['custom']['app_platform'] != "Android":
        continue #debug, only handle android actions
    visitor_data = {} #dict to store data for each user
    platform = visitor['custom']['app_platform']
    version = visitor['custom']['app_platform']
    visit_time = visitor['custom']['~d']
    visitor_number = visitor['vnum']
    user_events = [] #list of (unix time, event number) tuples

    for event in visitor['actions']:
        action = event['name']
        if action == 'property update':
            continue
        if not action:
            action = "None"
        '''
        if 'page_title' in event: #screen on app
            action = action + "- " + event['page_title']
        elif 'error-code' in event: #error
            action = action + "- " + event['error-message']
        '''
        if action not in action_dict:
            action_dict[action] = len(action_dict) #maps action to integer
        user_events.append((int(event['~m']), action_dict[action], action))

    visitor_data['platform'] = platform
    visitor_data['vnum'] = visitor_number
    visitor_data['version'] = version
    visitor_data['time'] = int(visit_time)
    visitor_data['events'] = {'count' : len(user_events), 'list' : user_events}
    visitors_data.append(visitor_data) #index by visit number


##todo: Apply Dataset Augmentation.
#dataset can be augmented with samples that contain error-related behavior.
#these error-related samples will be different enough from the normal Dataset
#so they can be differentiated via clustering


#start learning using KMeans clustering on data
#organize data from visitors_data into input array
'''
#features: event ID sequence
x = np.empty((1,11), int)
print(x.shape)
for __, value in visitors_data.items():
    act_IDs = np.ones((1,11)) * 999
    i = 0
    for event in value['events']['list']:
        act_IDs[0][i] = event[1]
        i += 1
    x = np.append(x, act_IDs, axis=0)
x = x[1:, :] #remove empty row
'''

'''
#features: visit time, event count, visit length, last action
x = np.empty((1,4), int)
print(x.shape)
for __, value in visitors_data.items():
    features = np.ones((1,4))
    features[0][0] = value['time'] #time of visit
    features[0][1] = value['events']['count'] #number of events
    features[0][2] = value['events']['list'][0][0] - value['time'] #time of visit
    features[0][3] = value['events']['list'][0][1]
    x = np.append(x, features, axis=0)
x = x[1:, :] #remove empty row

'''

'''
#features: construct histogram of ALL actions for each user as one feature
#  find most common action, average time between actions, number of actions
num_actions = len(action_dict)
x = np.empty((1,(num_actions + 3)), int)
for value in visitors_data:
    features = np.zeros((1,num_actions + 3)) #placeholder vector
    base_time = value['time']
    total_time = 0
    for user_action in value['events']['list']:
        features[0][user_action[1]] += 1 #if the action is present, add at index
        total_time += (user_action[0]- base_time)
    features[0][num_actions] = np.argmax(features) #most frequent action
    features[0][num_actions + 1] = value['events']['count'] #number of events
    features[0][num_actions + 2] = total_time / value['events']['count'] #average time between events
    x = np.append(x, features, axis=0)
x = x[1:, num_actions:] #remove empty row
'''

num_actions = 0
#features: above feature sans-histogram. avoid property-update as well
#  find most common action, average time between actions, number of actions
x = np.empty((1,(num_actions + 3)), int)
for value in visitors_data:
    actions = np.zeros((1, len(action_dict)))
    features = np.zeros((1, 3)) #placeholder vector
    base_time = value['time']
    total_time = 0
    for user_action in value['events']['list']:
        actions[0][user_action[1]] += 1 #if the action is present, add at index
        total_time += (user_action[0]- base_time)
    features[0][num_actions] = np.argmax(actions) #most frequent action
    features[0][num_actions + 1] = value['events']['count'] #number of events
    features[0][num_actions + 2] = total_time / value['events']['count'] #average time between events
    x = np.append(x, features, axis=0)
x = x[1:, num_actions:] #remove empty row

preprocessing.scale(x[:, num_actions+2], copy=False) #scales time column with zero mean
#learning parameters
clusters = 2

#potentially PCA, other features
reduced_data = x


###todo: set program to train and test from different datasets
##possibly make another API call once training complete
##move all of the data preprocessing to functions

#KMeans model
model = KMeans(n_clusters=clusters).fit(reduced_data)

#Gaussian Mixture model
#model = GaussianMixture(n_components=clusters).fit(x)

predictions = [] #predictions stored in this array. determines color in graph
i=0
for value in x:
    predictions.append(model.predict(value.reshape(1,-1))[0]) #predicts model for each point
    visitors_data[i]['cluster'] = int(predictions[i])
    i += 1
#saves data to external JSON for debug
outfile = open("output.json", 'w')
actions = open("actions.json", 'w')
resp_file = open("app_response.json", "w")
json.dump(action_dict, actions)
json.dump(visitors_data, outfile)
json.dump(resp, resp_file)


##places results in 3d graph
fig = plt.figure()
ax = Axes3D(fig)

#generate pallete for clustersplt.legend(handles=legend_handles)

cluster_color = [] #based on selected cluster
legend_handles = []
for i in range(clusters):
    color = (rand.uniform(0,0.7), rand.uniform(0,0.7), rand.uniform(0,0.7))
    cluster_color.append(color)
    legend_handles.append(mpatches.Patch(color=color, \
    label='Cluster {}'.format(i)))

#creates plot legend
plt.legend(handles=legend_handles)








num_actions = 0
###todo: adjust plot function to work with all feature selections in the future.
for i in range(0, len(x)): #plot each point + it's index as text above
    ax.scatter(x[i,num_actions],x[i,num_actions+1],x[i,num_actions+2],\
    c=cluster_color[predictions[i]])
    #ax.scatter(reduced_data[i,0],reduced_data[i,1],reduced_data[i,2],\
    #c=cluster_color[predictions[i]])

centroids = model.cluster_centers_

ax.scatter(centroids[:,num_actions],centroids[:,num_actions+1],centroids[:,num_actions+2],\
marker='x', s=169, linewidths=6,color='k', zorder=10)


plt.legend(handles=legend_handles)
ax.set_xlabel('most frequent action')
ax.set_ylabel('number of actions')
ax.set_zlabel('time between actions')
plt.show()
