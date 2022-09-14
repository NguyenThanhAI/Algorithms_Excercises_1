import numpy as np

n, k = str(input()).split(" ")

n = int(n)
k = int(k)

A = []

for _ in range(n):
    row = str(input()).split(" ")
    row = list(map(int, row))
    assert len(row) == n
    A.append(row)

A = np.array(A, dtype=np.int)

print(A)

G = np.zeros(shape=(n + 1, n + 1), dtype=np.int)

G[1, 1] = A[0, 0]
acc_sum = A[0, 0]
for j in range(2, n + 1):
    acc_sum += A[0, j - 1]
    G[1, j] = acc_sum

acc_sum = A[0, 0]
for i in range(2, n + 1):
    acc_sum += A[i - 1, 0]
    G[i, 1] = acc_sum

for i in range(2, n + 1):
    for j in range(2, n + 1):
        G[i, j] = G[i, j - 1] + G[i - 1, j] - G[i - 1, j - 1] + A[i - 1, j - 1]


print(G)

max_value = 0

for i in range(k, n + 1):
    for j in range(k, n + 1):
        value = G[i, j] - G[i - k, j] - G[i, j - k] + G[i - k, j - k]
        #print(i, j, value, G[i, j ] , G[i - k, j] , G[i, j - k] , G[i - k, j - k])
        if value > max_value:
            max_value = value
            max_row = i - k + 1
            max_col = j - k + 1

print(max_value)
print(max_row, max_col)