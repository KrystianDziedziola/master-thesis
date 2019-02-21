from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = [20, 5]


def draw_plots(input, target, headers, patient, ratio, end_second, start_second=0):
    patient_columns = input[patient]
    targets = target[patient]
    time_column = patient_columns[:, -1]
    channel_columns = patient_columns[:, :-1]
    (rows, columns) = channel_columns.shape
    
    for column in range(columns):
        channel = channel_columns[:, column]
        plt.figure(column)
        plt.xlabel('Seconds')
        plt.ylabel('Value')
        plt.title(headers[column])
        start = start_second * ratio
        end = end_second * ratio
        plt.plot(time_column[start:end], channel[start:end])
        draw_targets_in_range(targets, start_second, end_second)
     
    
def draw_targets_in_range(targets, start_second, end_second):
    for (target, target_index) in targets:
        if (target >= start_second) and (target <= end_second):
            plt.axvline(x=target, color='r')
            
            
def draw_plots_with_chunks(input, target, headers, patient, chunks_input, to_pdf=False):
    patient_columns = input[patient]
    targets = target[patient]
    time_column = patient_columns[:, -1]
    channel_columns = patient_columns[:, :-1]
    (rows, columns) = channel_columns.shape
    chunks_input = chunks_input[patient]
    
    for column in range(columns):
        channel = channel_columns[:, column]
        plt.figure(column)
        plt.xlabel('Seconds')
        plt.ylabel('Value')
        plt.title(headers[column])
        plt.plot(time_column, channel)
        draw_targets(targets)
        draw_chunks_start(chunks_input)
        if to_pdf:
            plt.savefig("chanel-{}.pdf".format(column), bbox_inches='tight')
      
    
def draw_targets(targets):
    for (target, target_index) in targets:
        plt.axvline(x=target, color='r')

        
def draw_chunks_start(chunks_input):
    for chunk_number in range(0, len(chunks_input)):
        chunk = chunks_input[chunk_number]
        
        plt.axvline(x=chunk[0, -1], color='b', linestyle="--")
        
        
