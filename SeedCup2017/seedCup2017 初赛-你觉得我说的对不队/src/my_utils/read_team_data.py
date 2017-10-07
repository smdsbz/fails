#!/usr/bin/env python3

'''
read team member data to a **defined data structure**
'''

__author__ = 'smdsbz'


import pandas as pd
# from itertools import izip

if __name__ == '__main__':
    TEAM_DATA_PATH = '../teamData.csv'
else:
    TEAM_DATA_PATH = './teamData.csv'

# global var
team_features = {}  # 'team_name': [ features ]



def read_team_data(path=TEAM_DATA_PATH):
    '''
    read team data, parse it to a dict, defined as follow
    @param:     TEAM_DATA_PATH
    @return:    team_features = {
                    str(team_name): [
                        feature_1,
                        feature_2,
                        ...
                    ],
                    ...
                }
    '''

    team_data = pd.read_csv(path)
    grouped_by_team_name = team_data.groupby('队名')

    team_features = {}
    for name, features in grouped_by_team_name:
            team_features[str(name)] = \
                modify_team_feature(features)

    return team_features




def modify_team_feature(team_feature):

    # 投篮
    def _iter_uptime_shot_ratio(team_feature):
        for time, ratio in zip(team_feature['上场时间'],
                               team_feature['投篮命中率']):
            if not isinstance(ratio, str):
                ratio = '0'
            yield time * float(ratio.replace('%', ''))

    def _iter_uptime_shots(team_feature):
        for time, shots in zip(team_feature['上场时间'],
                               team_feature['投篮出手次数']):
            yield time * shots

    def _iter_uptime_targets(team_feature):
        for time, target in zip(team_feature['上场时间'],
                                team_feature['投篮命中次数']):
            yield time * target

    # 罚球
    def _iter_uptime_bonus_rate(team_feature):
        for time, ratio in zip(team_feature['上场时间'],
                               team_feature['罚球命中率']):
            if not isinstance(ratio, str):
                ratio = '0'
            yield time * float(ratio.replace('%', ''))

    def _iter_uptime_bonus_shots(team_feature):
        for time, shots in zip(team_feature['上场时间'],
                                team_feature['罚球出手次数']):
            yield time * shots

    def _iter_uptime_bonus_targets(team_feature):
        for time, target in zip(team_feature['上场时间'],
                                team_feature['罚球命中次数']):
            yield time * target

    # 三分
    def _iter_uptime_three_rate(team_feature):
        for time, ratio in zip(team_feature['上场时间'],
                               team_feature['三分命中率']):
            if not isinstance(ratio, str):
                ratio = '0'
            yield time * float(ratio.replace('%', ''))

    def _iter_uptime_three_shots(team_feature):
        for time, shots in zip(team_feature['上场时间'],
                                team_feature['三分出手次数']):
            yield time * shots

    def _iter_uptime_three_targets(team_feature):
        for time, target in zip(team_feature['上场时间'],
                                team_feature['三分命中次数']):
            yield time * target

    # 篮板
    def _iter_uptime_boards(team_feature):
        for time, boards in zip(team_feature['上场时间'],
                                team_feature['篮板总数']):
            yield time * boards

    def _iter_uptime_boards_front(team_feature):
        for time, boards in zip(team_feature['上场时间'],
                                team_feature['前场篮板']):
            yield time * boards

    def _iter_uptime_boards_back(team_feature):
        for time, boards in zip(team_feature['上场时间'],
                                team_feature['后场篮板']):
            yield time * boards

    # 助攻
    def _iter_uptime_support(team_feature):
        for time, support in zip(team_feature['上场时间'],
                                team_feature['后场篮板']):
            yield time * support

    # 抢断
    def _iter_uptime_cut(team_feature):
        for time, cut in zip(team_feature['上场时间'],
                                team_feature['抢断']):
            yield time * cut

    # 盖帽
    def _iter_uptime_hat(team_feature):
        for time, hat in zip(team_feature['上场时间'],
                                team_feature['盖帽']):
            yield time * hat

    # 失误
    def _iter_uptime_fail(team_feature):
        for time, fail in zip(team_feature['上场时间'],
                                team_feature['失误']):
            yield time * fail

    # 犯规
    def _iter_uptime_foul(team_feature):
        for time, foul in zip(team_feature['上场时间'],
                                team_feature['犯规']):
            yield time * foul

    # 得分
    def _iter_uptime_score(team_feature):
        for time, score in zip(team_feature['上场时间'],
                                team_feature['得分']):
            yield time * score


    return [
        sum(_iter_uptime_shot_ratio(team_feature)) * 1e-4,
        sum(_iter_uptime_shots(team_feature)) * 1e-3,
        sum(_iter_uptime_targets(team_feature)) * 1e-2,
        sum(_iter_uptime_bonus_rate(team_feature)) * 1e-4,
        sum(_iter_uptime_bonus_shots(team_feature)) * 1e-2,
        sum(_iter_uptime_bonus_targets(team_feature)) * 1e-2,
        sum(_iter_uptime_three_rate(team_feature)) * 1e-3,
        sum(_iter_uptime_three_shots(team_feature)) * 1e-2,
        sum(_iter_uptime_three_targets(team_feature)) * 1e-2,
        sum(_iter_uptime_boards(team_feature)) * 1e-2,
        sum(_iter_uptime_boards_front(team_feature)) * 1e-2,
        sum(_iter_uptime_boards_back(team_feature)) * 1e-2,
        sum(_iter_uptime_support(team_feature)) * 1e-2,
        sum(_iter_uptime_cut(team_feature)) * 1e-2,
        sum(_iter_uptime_hat(team_feature)) * 1e-2,
        sum(_iter_uptime_fail(team_feature)) * 1e-2,
        sum(_iter_uptime_foul(team_feature)) * 1e-2,
        sum(_iter_uptime_score(team_feature)) * 1e-3
    ]



def _iter_over_teams(team_features=team_features):

    for name, features in team_features.items():
        yield features


if __name__ == '__main__':
    team_features = read_team_data()
    print(team_features['123'])
    # print(modify_team_feature(team_features['123']))
    # for team in team_features:
    #     print(modify_team_feature(team_features[team]))
    for name, features in team_features.items():
        print(name + ':', features)
