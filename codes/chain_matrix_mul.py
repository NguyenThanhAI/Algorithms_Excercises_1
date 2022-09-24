
'''import numpy as np

def print_result(trace, start, end):
    if start == end:
        print("A[{}]".format(start))
        return

    k = trace[start-1][end-1]
    print("(")
    print_result(trace, start, k)
    print_result(trace, k, end)
    print(")")

n = 5
seq = [4, 10, 3, 12, 20, 7]

L = np.zeros(shape=(len(seq) - 1, len(seq) - 1), dtype=np.int)
T = np.zeros(shape=(len(seq) - 1, len(seq) - 1), dtype=np.int)


for i in range(1, n):
    L[i - 1, i] = seq[i - 1] * seq[i] * seq[i + 1]

    T[i - 1, i] = i

#for i in range(1, n):

    for j in range(i + 2, n + 1):
        min_val = np.inf
        for k in range(i, j):
            val = L[i - 1, k - 1] + L[k - 1, j - 1] + seq[i - 1] * seq[k - 1] * seq[j - 1]
            if val < min_val:
                min_val = val
                bp = k
        L[i - 1, j - 1] = val
        T[i - 1, j - 1] = bp

print(L)
print(T)

#print_result(T, 1, n)'''
import numpy as np

def MatChainMul(arr, n):
    dp = [[0 for i in range(n)] for j in range(n)]
    t = np.zeros(shape=(n, n), dtype=np.int)
    for i in range(1, n+1):
        dp[i-1][i-1] = 0
    
    for L in range(1, n):
        for i in range(1, n-L+1):
            j = i+L
            dp[i - 1][j - 1] = 10**10
            print(i, j)
            for k in range(i, j):
                q = dp[i - 1][k - 1] + dp[k][j - 1] + arr[i-1]*arr[k]*arr[j]
                if q < dp[i - 1][j - 1]:
                    dp[i - 1][j - 1] = q
                    t[i - 1][j - 1] = k
    print(np.array(dp), t)
    return dp[0][n-1]

arr = [4, 10, 3, 12, 20, 7]
size = len(arr) - 1

print("Minimum number of multiplications are " + str(MatChainMul(arr, size)))

'''import numpy as np

def matrix_product(p):
    """Return m and s.
 
    m[i][j] is the minimum number of scalar multiplications needed to compute the
    product of matrices A(i), A(i + 1), ..., A(j).
 
    s[i][j] is the index of the matrix after which the product is split in an
    optimal parenthesization of the matrix product.
 
    p[0... n] is a list such that matrix A(i) has dimensions p[i - 1] x p[i].
    """
    length = len(p) # len(p) = number of matrices + 1
 
    # m[i][j] is the minimum number of multiplications needed to compute the
    # product of matrices A(i), A(i+1), ..., A(j)
    # s[i][j] is the matrix after which the product is split in the minimum
    # number of multiplications needed
    m = [[-1]*length for _ in range(length)]
    s = [[-1]*length for _ in range(length)]
 
    matrix_product_helper(p, 1, length - 1, m, s)
 
    return m, s
 
 
def matrix_product_helper(p, start, end, m, s):
    """Return minimum number of scalar multiplications needed to compute the
    product of matrices A(start), A(start + 1), ..., A(end).
 
    The minimum number of scalar multiplications needed to compute the
    product of matrices A(i), A(i + 1), ..., A(j) is stored in m[i][j].
 
    The index of the matrix after which the above product is split in an optimal
    parenthesization is stored in s[i][j].
 
    p[0... n] is a list such that matrix A(i) has dimensions p[i - 1] x p[i].
    """
    if m[start][end] >= 0:
        return m[start][end]
 
    if start == end:
        q = 0
    else:
        q = float('inf')
        for k in range(start, end):
            temp = matrix_product_helper(p, start, k, m, s) \
                   + matrix_product_helper(p, k + 1, end, m, s) \
                   + p[start - 1]*p[k]*p[end]
            if q > temp:
                q = temp
                s[start][end] = k
 
    m[start][end] = q
    return q
 
 
def print_parenthesization(s, start, end):
    """Print the optimal parenthesization of the matrix product A(start) x
    A(start + 1) x ... x A(end).
 
    s[i][j] is the index of the matrix after which the product is split in an
    optimal parenthesization of the matrix product.
    """
    if start == end:
        print('A[{}]'.format(start), end='')
        return
 
    k = s[start][end]
 
    print('(', end='')
    print_parenthesization(s, start, k)
    print_parenthesization(s, k + 1, end)
    print(')', end='')
 
 
n = int(input('Enter number of matrices: '))
p = []
for i in range(n):
    temp = int(input('Enter number of rows in matrix {}: '.format(i + 1)))
    p.append(temp)
temp = int(input('Enter number of columns in matrix {}: '.format(n)))
p.append(temp)
 
m, s = matrix_product(p)
print(np.array(s))
print('The number of scalar multiplications needed:', m[1][n])
print('Optimal parenthesization: ', end='')
print_parenthesization(s, 1, n)'''