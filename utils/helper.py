import numpy as np
import os

def display_non_zero(array, top_k):
    # array is a 2D numpy array
    non_zero_entries = np.nonzero(array)
    print("Number of non-zero entries: {}".format(len(non_zero_entries[0])))

    # sort non-zero entries by value
    non_zero_entries = zip(*sorted(zip(*non_zero_entries), key=lambda x: array[x[0]][x[1]], reverse=True))
    
    k = 0
    print("Top", top_k, "Non-zero entries:")
    for i, j in zip(*non_zero_entries):
        if k >= top_k:
            break
        print("Row: {}, Column: {}, Value: {}".format(i, j, array[i][j]))
        k += 1

def action2elec_amp(action_idx, n_amps):
    # convert action_idx to (elec, amp)
    (elec, amp) = (action_idx//n_amps + 1, action_idx%n_amps + 1)
    return (elec, amp)

def elec_amp2action(elec, amp, n_amps):
    # convert (elec, amp) to action_idx
    action_idx = (elec - 1)*n_amps + (amp - 1)
    return action_idx

def get_dict_from_action_idx(action_idx, n_amps, path):
    # convert action_idx to (elec, amp)
    (elec, amp) = action2elec_amp(action_idx, n_amps)
    
    dict, elecs, amps, elec_map, cell_ids = load_dictionary(path)

    # get dictionary entry
    try:
        idx = np.where((elecs == elec) & (amps == amp))[0][0]
        dict_entry = dict[idx]
        found = True
    except IndexError:
        # print(f"Electrode {elec} with amplitude {amp} was not in the dictionary")
        # print(f"Assume no cells were activated")
        dict_entry = np.zeros(len(cell_ids), dtype=np.float64)
        found = False

    return dict_entry, found

def load_dictionary(path):
    # Load relevant data from .npz files
    try:
        with np.load(os.path.join(path,"dictionary.npz")) as data:
            dict = data["dictionary"]
            elecs = data["entry_elecs"]
            amps = data["entry_amps"]
            elec_map = data["elec_map"]
        with np.load(os.path.join(path,"decoders.npz")) as data:
            cell_ids = data["cell_ids"]
    except FileNotFoundError:
        print("Please make sure the dictionary.npz and decoders.npz files are present in the specified path")

    return dict, elecs, amps, elec_map, cell_ids