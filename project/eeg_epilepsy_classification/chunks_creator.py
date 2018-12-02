import random
import numpy as np


def create_chunks_with_seizures(patient_data, seizure_seconds, chunk_size):
    number_of_chunks = len(seizure_seconds)
    #     16 when without time
    chunks_input = np.zeros((number_of_chunks, chunk_size, 17))
    chunks_target = np.zeros(number_of_chunks)

    for seizure_number in range(0, number_of_chunks):
        (seizure_time, seizure_index) = seizure_seconds[seizure_number]
        chunk_start_index = seizure_index
        chunk_end_index = chunk_start_index + chunk_size
        chunks_input[seizure_number] = patient_data[chunk_start_index:chunk_end_index, :]
        # seizure
        chunks_target[seizure_number] = 1

    return (chunks_input, chunks_target)


def is_in_seizure_range(index, seizure_seconds, chunk_size):
    for (seizure_time, seizure_index) in seizure_seconds:
        seizure_start_index = seizure_index
        seizure_end_index = seizure_start_index + chunk_size
        if index in range(seizure_start_index, seizure_end_index):
            return True

    return False


def create_non_seizure_data_start_index(data_size, chunk_size, seizure_seconds):
    start_index = random.randint(0, data_size - chunk_size)

    while (is_in_seizure_range(start_index, seizure_seconds, chunk_size)):
        start_index = random.randint(0, data_size - chunk_size)

    return start_index


def create_chunks_without_seizures(patient_data, seizure_seconds, chunk_size):
    number_of_chunks = len(seizure_seconds)
    #     16 when without time
    chunks_input = np.zeros((number_of_chunks, chunk_size, 17))
    chunks_target = np.zeros(number_of_chunks)
    (data_size, channels) = patient_data.shape

    for chunk_number in range(0, number_of_chunks):
        chunk_start_index = create_non_seizure_data_start_index(data_size, chunk_size, seizure_seconds)

        chunk_end_index = chunk_start_index + chunk_size
        chunks_input[chunk_number] = patient_data[chunk_start_index:chunk_end_index, :]
        # non-seizure
        chunks_target[chunk_number] = 0

    return (chunks_input, chunks_target)


def flatten_chunks(chunks_input, chunks_target):
    train_input = []
    train_target = []

    for patient_number in range(0, len(chunks_input)):
        patient_data = chunks_input[patient_number]
        patient_targets = chunks_target[patient_number]
        for chunk_number in range(0, len(patient_data)):
            train_input.append(patient_data[chunk_number])
            train_target.append(patient_targets[chunk_number])

    train_input = np.array(train_input)
    train_target = np.array(train_target)

    #remove time column
    train_input = train_input[:, :, :-1]
    
    return train_input, train_target 


def prepare_chunks(input, target, chunk_size_in_seconds, ratio):
    chunk_size = chunk_size_in_seconds * ratio
    chunks_input = []
    chunks_target = []

    for patient_number in range(0, len(input)):
        patient_chunks_input = []
        patient_chunks_target = []
        seizure_seconds = target[patient_number]
        patient_data = input[patient_number]
        (seizure_chunks_input, seizure_chunks_target) = create_chunks_with_seizures(patient_data,
                                                                                    seizure_seconds,
                                                                                    chunk_size)
        patient_chunks_input.extend(seizure_chunks_input)
        patient_chunks_target.extend(seizure_chunks_target)

        (non_seizure_chunks_input, non_seizure_chunks_target) = create_chunks_without_seizures(patient_data,
                                                                                               seizure_seconds,
                                                                                               chunk_size)
        patient_chunks_input.extend(non_seizure_chunks_input)
        patient_chunks_target.extend(non_seizure_chunks_target)

        chunks_input.append(np.array(patient_chunks_input))
        chunks_target.append(np.array(patient_chunks_target))

    return np.array(chunks_input), np.array(chunks_target)
