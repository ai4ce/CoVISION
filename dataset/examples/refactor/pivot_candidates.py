import numpy as np
from numba import njit

# @TODO: check numba
def pivot_candidates(original_map: np.ndarray) -> np.ndarray:
    """
    Returns a NumPy array of pivot candidates from the original_map.
    Returned pivot array is a list of [x,y,corner] where corner is in [0..7] inclusive.
    1: empty
    0: occupied
    
    """
    pivot_array = []
    rows, cols = original_map.shape
    for i in range(rows):
        for j in range(cols):
            if original_map[i][j] == 1:
                is_top_row = i - 1 < 0
                is_bottom_row = i + 1 > rows - 1
                is_left_col = j - 1 < 0
                is_right_col = j + 1 > cols - 1
                if is_top_row:
                    if is_left_col:
                        if (
                            original_map[i][j + 1] == 0
                            or original_map[i + 1][j] == 0
                        ):
                            pivot_array.append([i, j])
                    elif is_right_col:
                        if (
                            original_map[i][j - 1] == 0
                            or original_map[i + 1][j] == 0
                        ):
                            pivot_array.append([i, j])
                    else:
                        if (
                            original_map[i + 1][j] == 0
                            or original_map[i][j - 1] == 0
                        ):
                            pivot_array.append([i, j])
                elif is_bottom_row:
                    if is_left_col:
                        if(
                            original_map[i][j + 1] == 0
                            or original_map[i - 1][j] == 0
                        ):
                            pivot_array.append([i, j])
                    elif is_right_col:
                        if (
                            original_map[i][j - 1] == 0
                            or original_map[i - 1][j] == 0
                        ):
                            pivot_array.append([i, j])
                    else:
                        if (
                            original_map[i - 1][j] == 0
                            or original_map[i][j - 1] == 0
                            or original_map[i][j + 1] == 0
                        ):
                            pivot_array.append([i, j])
                else:
                    if is_left_col:
                        if (
                            original_map[i][j + 1] == 0
                            or original_map[i + 1][j] == 0
                            or original_map[i - 1][j] == 0
                        ):
                            pivot_array.append([i, j])
                    elif is_right_col:
                        if (
                            original_map[i + 1][j] == 0
                            or original_map[i - 1][j] == 0
                            or original_map[i][j - 1] == 0
                        ):
                            pivot_array.append([i, j])
                    else:
                        # corners = ['lu', 'ld', 'ru', 'rd', 'l', 'u', 'r', 'd']
                        if original_map[i - 1][j] == 0:
                            if original_map[i][j - 1] == 0:
                                pivot_array.append([i, j, 0])
                            elif original_map[i][j + 1] == 0:
                                pivot_array.append([i, j, 2])
                            else:
                                pivot_array.append([i, j, 5])
                        elif original_map[i + 1][j] == 0:
                            if original_map[i][j - 1] == 0:
                                pivot_array.append([i, j, 1])
                            elif original_map[i][j + 1] == 0:
                                pivot_array.append([i, j, 3])
                            else:
                                pivot_array.append([i, j, 7])
                        elif original_map[i][j - 1] == 0:
                            pivot_array.append([i, j, 4])
                        elif original_map[i][j + 1] == 0:
                            pivot_array.append([i, j, 6])
                        # check if any of the 8 neighbors are 0
                        elif (
                            original_map[i - 1][j - 1] == 0
                            or original_map[i - 1][j + 1] == 0
                            or original_map[i + 1][j - 1] == 0
                            or original_map[i + 1][j + 1] == 0
                        ):
                            pivot_array.append([i, j, 7])
    return np.array(pivot_array,dtype=int)  # pivot array is a list of [x,y,corner] where corner is 0,1,2,3,4,5,6,7