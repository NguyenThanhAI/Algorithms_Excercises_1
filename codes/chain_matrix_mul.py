import numpy as np

n = int(input())

arr = str(input()).split(" ")
arr = list(map(int, arr))
assert len(arr) == n + 1

def printParens(t, i, j):
    if i == j:
        print("A_{}".format(i), end="")

    else:
        print("(", end="")
        printParens(t, i, t[i, j])
        printParens(t, t[i , j] + 1, j)
        print(")", end="")


dp = [[0 for i in range(n)] for j in range(n)]
t = np.zeros(shape=(n+1, n+1), dtype=np.int)
for i in range(1, n+1):
    dp[i-1][i-1] = 0
    
for L in range(1, n):
    for i in range(1, n-L+1):
        j = i+L
        dp[i - 1][j - 1] = np.inf
        #print(i, j)
        for k in range(i, j):
            q = dp[i - 1][k - 1] + dp[k][j - 1] + arr[i-1]*arr[k]*arr[j]
            if q < dp[i - 1][j - 1]:
                dp[i - 1][j - 1] = q
                t[i][j] = k
#print(np.array(dp), t)

#arr = [4, 10, 3, 12, 20, 7]


print(dp[0][n-1])
#print(t)
printParens(t, 1, n)