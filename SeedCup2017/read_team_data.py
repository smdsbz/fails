#!/usr/bin/env python3

'''
loads team member data to a defined data structure
'''


__author__ = 'smdsbz'


import pandas as pd


TEAM_DATA_PATH = './teamData.csv'


def load_team_data(path):
    
    data = pd.read_csv(path)
    team_total = data.iloc[-1][0]
    for teams in range(team_total):

