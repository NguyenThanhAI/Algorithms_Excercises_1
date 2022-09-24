
import numpy as np

#a = "abcdaabcdaab"
#b = "adehaabcdaaeios"

a = str(input())
b = str(input())


S = np.zeros(shape=(len(a)+1, len(b)+1))

max_len = - np.inf

for i in range(1, len(a) + 1):
    for j in range(1, len(b) + 1):
        if a[i - 1] == b[j - 1]:
            S[i, j] = 1 + S[i - 1, j - 1]
            if S[i, j] > max_len:
                max_len = S[i, j]
                pos = (i, j)
        else:
            S[i, j] = 0


#print(S)
#print(pos - max_len + 1)
#print(pos[0])
print(a[int(pos[0] - max_len): int(pos[0])])