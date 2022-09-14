import numpy as np

N, L = str(input()).split(" ")
N = int(N)
L = int(L)

w = []
t = []

for _ in range(N):
    x, y = str(input()).split(" ")
    x = int(x)
    y = int(y)
    w.append(x)
    t.append(y)

print(N, L)
print(w)
print(t)

w = np.array(w)
t = np.array(t)

C = [0]

trace = []
#bp = 0
for i in range(1, N+1):
    temp_c = np.inf
    split = False
    for k in range(1, i+1):
        if np.sum(w[k-1:i]) <= L:
            if temp_c > C[k-1] + np.max(t[k-1:i]):
                temp_c = C[k-1] + np.max(t[k-1:i])
                split = True
                bp = k
    if split:
        trace.append(bp)
    C.append(temp_c)

bps = list(set(trace))

print(C[N])

for pt in bps:
    print(w[pt - 1], t[pt - 1])