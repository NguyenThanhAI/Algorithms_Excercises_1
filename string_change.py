import numpy as np

orig_string = input()
target_string = input()

L = np.zeros(shape=(len(orig_string) + 1, len(target_string) + 1), dtype=np.int)


for i in range(len(orig_string) + 1):
    for j in range(len(target_string) + 1):
        if i == 0:
            L[i, j] = j
            continue

        if j == 0:
            L[i, j] = i
            continue

        if orig_string[i-1] == target_string[j-1]:
            L[i, j] = L[i - 1, j - 1]
        else:
            L[i, j] = min([L[i-1, j], L[i, j-1], L[i-1, j-1]]) + 1


print(L[len(orig_string), len(target_string)])

print(L)

row = L.shape[0] - 1
col = L.shape[1] - 1

trace = []

while row != 0 or L[row, col] != 0:
    if L[row, col] == L[row, col-1] + 1:
        trace.append("add")
        col = col - 1
    elif L[row, col] == L[row-1, col-1]:
        trace.append("none")
        row = row - 1
        col = col - 1
    elif L[row, col] == L[row - 1, col - 1] + 1:
        trace.append("adjust")
        row = row - 1
        col = col -1
    elif L[row, col] == L[row-1, col] + 1:
        trace.append("remove")
        row = row - 1

#trace.reverse()
    
print(trace)

trace = []

'''def backtrack(row: int, col: int):

    for delta_row, delta_col in [(-1, 0), (-1, -1), (0, -1)]:
        if delta_row == -1 and delta_col == -1:
            if'''