import numpy as np

'''rates = [
    [1, 0.23, 0.25, 16.43, 18.21, 4.94],
    [4.34, 1, 1.11, 71.40, 79.09, 21.44],
    [3.93, 0.90, 1, 64.52, 71.48, 19.37],
    [0.061, 0.014, 0.015, 1, 1.11, 0.30],
    [0.055, 0.013, 0.014, 0.90, 1, 0.27],
    [0.20, 0.047, 0.052, 3.33, 3.69, 1],
]
'''

n = int(input())

R = []

for _ in range(n):
    row = str(input()).split(" ")
    row = list(map(float, row))
    assert len(row) == n
    R.append(row)

R = np.array(R, dtype=np.float32)

print(R)

D = np.zeros_like(R, dtype=np.float32)
T = np.zeros_like(D, dtype=np.int)

for i in range(n):
    for j in range(n):

        D[i, j] = - np.log(R[i, j])
        T[i, j] = j
        

print(D)
print(T)



for i in range(n):
    for j in range(n):
        for k in range(n):        
            #if k == i or k == j:
            #    continue
            if D[i, j] > D[i, k] + D[k, j]:
                D[i, j] = D[i, k] + D[k, j]
                #print(i, j, k, T[i, j], T[i, k])
                T[i, j] = T[i, k]

print(D)
minCycle = np.inf
for i in range(n):
    if D[i, i] < minCycle:
        minCycle = D[i, i]
        unit = i

print(unit)
print(T)
if minCycle < 0:
    #profit = np.exp(-minCycle)
    trace = []
    presentCurrency = unit
    trace.append(presentCurrency)
    while True:
        presentCurrency = T[presentCurrency, unit]
        trace.append(presentCurrency)
        #print(presentCurrency, unit)
        if presentCurrency == unit:
            profit = 1
            for i in range(len(trace) - 1):
                profit *= R[trace[i], trace[i + 1]]
            profit -= 1
            break

    print("YES")
    print(unit, profit)
    print(trace)
else:
    print("NO")