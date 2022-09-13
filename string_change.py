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


print(L)