import pandas as pd
import os
import numpy as np

class Accumulator:
    def __init__(self):
        self.total = 0
        self.count = 0
        self.items = []

    def add(self, x):
        if self.total != 0:
            self.total += x
        else:
            self.total = x
        self.count += 1
        self.items.append(x)

    def get_features(self):
        return {
            "avg": self.total / self.count,
            "percentile_25": np.percentile(self.items, 25),
            "percentile_50": np.percentile(self.items, 50),
            "percentile_75": np.percentile(self.items, 75),
        }


def extract_features2(data):
    down_down = Accumulator()
    up_down = Accumulator()
    dwell = Accumulator()

    last_key_event = {}
    down_down_unmatched_down = None
    up_down_unmatched_up = None

    for key, time, down in zip(data['keys'], data['time'], data['pressed']):
        if down:
            if down_down_unmatched_down is not None:
                down_down.add((time - down_down_unmatched_down).total_seconds())
                down_down_unmatched_down = None

            if up_down_unmatched_up is not None:
                up_down.add((time - up_down_unmatched_up).total_seconds())
                up_down_unmatched_up = None

            down_down_unmatched_down = time
        else:
            up_down_unmatched_up = time

            if key in last_key_event:
                dwell.add((time - last_key_event[key]).total_seconds())

        last_key_event[key] = time

    return down_down.get_features()['percentile_25'], down_down.get_features()['percentile_50'], dwell.get_features()['avg']


def extract_features(file_path):
    """
    Extract features from a single keylogger file.
    """
    df = pd.read_csv(file_path, header=None, names=['keys', 'time', 'pressed'])
    df['time'] = pd.to_datetime(df['time'], unit='s')

    PP_values, RR_values, PR_values, RP_values = [], [], [], []

    for i in range(1, len(df)):
        if df['pressed'][i - 1] and df['pressed'][i]:
            PP_values.append((df['time'][i] - df['time'][i - 1]).total_seconds())
        elif not df['pressed'][i - 1] and not df['pressed'][i]:
            RR_values.append((df['time'][i] - df['time'][i - 1]).total_seconds())
        elif df['pressed'][i - 1] and not df['pressed'][i]:
            PR_values.append((df['time'][i] - df['time'][i - 1]).total_seconds())
        elif not df['pressed'][i - 1] and df['pressed'][i]:
            RP_values.append((df['time'][i] - df['time'][i - 1]).total_seconds())

    avg_PP = sum(PP_values) / len(PP_values) if PP_values else 0
    avg_RR = sum(RR_values) / len(RR_values) if RR_values else 0
    avg_PR = sum(PR_values) / len(PR_values) if PR_values else 0
    avg_RP = sum(RP_values) / len(RP_values) if RP_values else 0

    down_down_25, down_down_50, avg_dwell = extract_features2(df)

    return avg_PP, avg_RR, avg_PR, avg_RP, down_down_25, down_down_50, avg_dwell


def extract_on_directory(directory_path):
    """
    Extract features from all keylogger files in a directory.
    """
    results_df = pd.DataFrame(columns=['Average_PP', 'Average_RR', 'Average_PR', 'Average_RP', 'Average_dwell', '25P_DownDown', '50P_DownDown', 'User_Label'])
    user_label_mapping = {}

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            avg_PP, avg_RR, avg_PR, avg_RP, down_down_25, down_down_50, avg_dwell = extract_features(file_path)
            username = filename.split('_')[1]

            if username not in user_label_mapping:
                user_label_mapping[username] = len(user_label_mapping)

            User_Label = user_label_mapping[username]

            results_df = results_df.append({
                'Average_PP': avg_PP,
                'Average_RR': avg_RR,
                'Average_PR': avg_PR,
                'Average_RP': avg_RP,
                'Average_dwell': avg_dwell,
                '25P_DownDown': down_down_25,
                '50P_DownDown': down_down_50,
                'User_Label': User_Label
            }, ignore_index=True)

    results_df.to_csv("train.csv", index=False)


def extract_live(csv_path):
    """
    Extract features from a single live keylogger CSV file.
    """
    results_df = pd.DataFrame(columns=['Average_PP', 'Average_RR', 'Average_PR', 'Average_RP', 'Average_dwell', '25P_DownDown', '50P_DownDown'])
    user_label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20}

    avg_PP, avg_RR, avg_PR, avg_RP, down_down_25, down_down_50, avg_dwell = extract_features(csv_path)
    username = csv_path.split('_')[1]
    supposed_prediction = user_label_mapping[username]

    results_df = results_df.append({
        'Average_PP': avg_PP,
        'Average_RR': avg_RR,
        'Average_PR': avg_PR,
        'Average_RP': avg_RP,
        'Average_dwell': avg_dwell,
        '25P_DownDown': down_down_25,
        '50P_DownDown': down_down_50
    }, ignore_index=True)

    results_df.to_csv("live_features.csv", index=False)
    return supposed_prediction
