import numpy as np

def display_non_zero(array):
    # array is a 2D numpy array
    non_zero_entries = np.nonzero(array)
    print("Number of non-zero entries: {}".format(len(non_zero_entries[0])))

    # sort non-zero entries by value
    non_zero_entries = zip(*sorted(zip(*non_zero_entries), key=lambda x: array[x[0]][x[1]], reverse=True))
    
    print("Non-zero entries:")
    for i, j in zip(*non_zero_entries):
        print("Row: {}, Column: {}, Value: {}".format(i, j, array[i][j]))