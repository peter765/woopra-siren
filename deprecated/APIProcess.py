'''
APIProcess.py
This is a python class designed to interface with the Woopra Online List API.

The Online List API is used to get a list of current users on the
woopra-monitored site, along with their device info and the list of actions per
visit(capped at 11 visits per record)

The APIProcess class contains methods to save the response(raw and processed)
to a file and reconnect to Woopra to pull more data.

In the future, it might be helpful to filter data based on OS and other
criteria, but this can also be handled by a database driver?

NOTES:
-ignores actions on iOS/Windows phone due to duplicate actions
-ignores property-update(potential noise)
},...]
'''
import json
import requests
import WoopraKeys
from sortedcontainers import SortedDict

#json parameters for HTTP POST request
payload = WoopraKeys.payload
api_url = WoopraKeys.api_url
api_auth = WoopraKeys.api_auth
uri = WoopraKeys.mongo_uri

client = MongoClient(uri)
visits = client['api-process'].visits



class APIProcess(object):
    def actionSort(self):
        i=0
        for key, v in self.actions.items():
            self.actions[key] = i
            i += 1
        for visitor in self.visitors:
            for event in visitor['events']['list']:
                try:
                    event[2] = self.actions[event[3]]
                    event[4] = self.actions[event[5]]
                except KeyError:
                    import pdb; pdb.set_trace()


    def actionParse(self, event):
        action = event['name']
        if action == "":
            action = "None"
        if 'page_title' in event: #screen on app
            action = action + "- " + event['page_title']
        elif 'error-code' in event: #error
            action = action + "- " + event['error-message']
        if action not in self.actions:
            self.actions[action] = len(self.actions) #maps action to integer
        return action

    def writeToFile(self):
        json.dump(self.visitors, self.outfile)
        json.dump(dict(self.actions), self.actfile)
        json.dump(self.resp, self.resp_file)

    def connect(self):
        #request data from server
        req = requests.post(api_url, auth=api_auth, params=payload)
        #encodes results as json
        try:
            self.resp = req.json()
        except JSONDecodeError:
            print("Unable to connect to network")
            exit()
        i = 0
        for visitor in self.resp['visitors']:
            if visitor['custom']['app_platform'] != "Android":
                continue #debug, only handle android actions
            visitor_data = {} #dict to store data for each user
            platform = visitor['custom']['app_platform']
            version = visitor['custom']['app_platform']
            visit_time = visitor['custom']['~d']
            visitor_number = visitor['vnum']
            user_events = [] #list of (unix time, event number) tuples
            i = 1;
            for event in visitor['actions']:
                action = self.actionParse(event)
                next_action = 'None'
                elapsed_time = 0
                if i < len(visitor['actions']):
                    next_action = self.actionParse(visitor['actions'][i])
                    if next_action is False:
                        continue
                    elapsed_time = int(event['~m']) - \
                    int(visitor['actions'][i]['~m'])
                    i += 1
                #append timestamp(ms), elapsed_time, act#, action, n_axt#, next_action
                try:
                    user_events.append([int(event['~m']), \
                    elapsed_time, self.actions[action],action, \
                    self.actions[next_action], next_action])
                except KeyError:
                    user_events.append([int(event['~m']), \
                    elapsed_time, self.actions["None"],action, \
                    self.actions[next_action], next_action])

            visitor_data['platform'] = platform
            visitor_data['vnum'] = visitor_number
            visitor_data['version'] = version
            visitor_data['time'] = int(visit_time)
            visitor_data['events'] = {'count' : len(user_events), \
            'list' : user_events}
            self.visitors.append(visitor_data) #index by visit number

    def __init__(self, site='mobile'):
        self.resp = ""
        self.actions = SortedDict()
        self.actions["None"] = len(self.actions)
        self.visitors = []
        filename = "logs/app_response.json"
        if site == 'web':
            payload = {'website' : 'newcricketwireless.com'}
            filename = "logs/web_response.json"
        elif site == 'amss':
            payload = {'website' : 'amss.at.cricketwireless.com'}
            filename = "logs/amss_response.json"
        self.outfile = open("logs/output.json", 'w')
        self.actfile = open("logs/actions.json", 'w')
        self.resp_file = open(filename, "w")
        self.connect()
        self.actionSort()
        result = visits.insert_many(dict(self.visitors))
