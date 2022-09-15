import numpy as np


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

minCycle = np.inf
for i in range(n):
    if D[i, i] < minCycle:
        minCycle = D[i, i]
        unit = i

if minCycle < 0:
    #profit = np.exp(-minCycle)
    trace = []
    presentCurrency = unit
    trace.append(presentCurrency + 1)
    while True:
        presentCurrency = T[presentCurrency, unit]
        trace.append(presentCurrency + 1)
        #print(presentCurrency, unit)
        if presentCurrency == unit:
            profit = 1
            for i in range(len(trace) - 1):
                profit *= R[trace[i] - 1, trace[i + 1] - 1]
            profit -= 1
            break

    print("YES")
    print(unit + 1, profit)
    for el in trace:
        print("{} ".format(el), end="")
else:
    print("NO")