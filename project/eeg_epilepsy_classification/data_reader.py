import os
import numpy as np
import pickle

from chunks_creator import prepare_chunks
from chunks_creator import flatten_chunks

from sklearn.preprocessing import StandardScaler

INPUT_DATA_FILE_PATH='tmp/input.pckl'

DATA_FREQUENCY = 500
SAMPLING_RATE = 5
FREQUENCY_TO_SAMPLING_RATIO = DATA_FREQUENCY // SAMPLING_RATE
CHUNK_SIZE_IN_SECONDS = 4


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


def load_data_to_file():
    (input_data, target, headers) = read_data(data_path='data', 
                                              sampling_rate=SAMPLING_RATE, 
                                              data_frequency=DATA_FREQUENCY)

    with open(INPUT_DATA_FILE_PATH, 'wb') as input_variable_file:
        pickle.dump([input_data, target, headers], input_variable_file)

    del input_data, target, headers
    
    
def normalize(x, y):
    scalers = {}
    for channel_number in range(x.shape[1]):
        scalers[channel_number] = StandardScaler()
        x[:, channel_number, :] = scalers[channel_number].fit_transform(x[:, channel_number, :]) 
    return x, y.astype(int)


def load_input_data():
    with open(INPUT_DATA_FILE_PATH, 'rb') as input_data_file:
        input_data, target, headers = pickle.load(input_data_file)
    
    return input_data, target, headers


def prepare_data(chunk_size_in_seconds):
    input_data, target, headers = load_input_data()
    
    chunks_input, chunks_target = prepare_chunks(input_data, 
                                                target, 
                                                chunk_size_in_seconds=chunk_size_in_seconds, 
                                                ratio=FREQUENCY_TO_SAMPLING_RATIO)
    x, y = flatten_chunks(chunks_input, chunks_target)
    x, y = normalize(x, y)
    
    return x, y


def get_data(chunk_size_in_seconds, load_from_sources = False):
    if (load_from_sources):
        load_data_to_file()
        
    return prepare_data(chunk_size_in_seconds)