import os
import numpy as np


def parse_file(file, sampling_rate):
    lines = file.split('\n')
    headers = lines[0].split('\t')
    # to one before last because the last one is empty
    data = lines[1:-1]

    number_of_lines = len(data)

    float_data = np.zeros((number_of_lines, len(headers)))
    for line_number, line in enumerate(data):
        values = [float(value) for value in line.split('\t')]
        float_data[line_number, :] = values

    return float_data[::sampling_rate], headers


def read_input_files(end, data_path, sampling_rate):
    input_path = os.path.join(data_path, 'input_500Hz/sick')
    input_file_names = os.listdir(input_path)
    input_file_names.sort(key=int)

    start = None

    files_content = []
    for file_name in input_file_names[start:end]:
        file_path = os.path.join(input_path, file_name)
        file = open(file_path, 'r')
        (columns, headers) = parse_file(file.read(), sampling_rate)
        print('Loaded input file:', file_name)
        file.close()
        files_content.append(columns)
    print('--Input files loaded--')
    return files_content, headers


def create_target_index(value, frequency_to_sampling_ratio):
    value = int(value)
    return int(value * frequency_to_sampling_ratio)


def read_target_files(end, data_path, sampling_rate, data_frequency):
    frequency_to_sampling_ratio = data_frequency // sampling_rate
    targets_path = os.path.join(data_path, 'targets')
    targets_file_name = os.listdir(targets_path)[0]
    targets_file_path = os.path.join(targets_path, targets_file_name)

    file = open(targets_file_path, 'r')
    targets_content = file.read()
    file.close()

    # last line is empty
    lines = targets_content.split('\n')[:-1]
    targets = []
    for number, line in enumerate(lines, 1):
        targets.append([(int(value), create_target_index(value, frequency_to_sampling_ratio)) for value in line.split(',')])
    print('--Target files loaded--')
    return targets[:end]


def read_data(data_path, sampling_rate, data_frequency, end=104):
    (input_data, headers) = read_input_files(end, data_path, sampling_rate)
    targets_data = read_target_files(end, data_path, sampling_rate, data_frequency)

    return input_data, targets_data, headers