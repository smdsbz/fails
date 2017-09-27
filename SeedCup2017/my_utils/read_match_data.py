#!/usr/bin/env python3

'''
read match data, parse to train features
'''

__author__ = 'smdsbz'

import pandas as pd

if __name__ == '__main__':
    MATCH_RESULT_PATH = '../matchDataTrain.csv'
else:
    MATCH_RESULT_PATH = './matchDataTrain.csv'
    TEST_DATA_PATH = './matchDataTest.csv'



match_results = [] # ( team_1, team_2, weighted_feature )


def read_match_data(path):

    match_data = pd.read_csv(path)

    # global match_results
    match_results = []
    for line in match_data.iterrows():

        guest_team = line[1]['客场队名']
        host_team = line[1]['主场队名']
        match_result = [ int(s) for s in line[1]['比分'].split(':') ]

        match_result_modified = \
            (match_result[0] - match_result[1]) / (match_result[0] + match_result[1])

        match_results.append([
            str(host_team),
            str(guest_team),
            match_result_modified * 1e+2
        ])

    return match_results


if __name__ == '__main__':
    match_results = read_match_data(path='../matchDataTest.csv')
    print(len(match_results))
    for result in match_results:
        print(result)
