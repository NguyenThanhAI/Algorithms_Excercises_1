
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

def print_parens(t, i, j):
    if i == j:
        print("A_{}".format(i), end="")

    else:
        print("(", end="")
        print_parens(t, i, t[i, j])
        print_parens(t, t[i , j] + 1, j)
        print(")", end="")


def MatChainMul(arr, n):
    dp = [[0 for i in range(n)] for j in range(n)]
    t = np.zeros(shape=(n+1, n+1), dtype=np.int)
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
                    t[i][j] = k
    print(np.array(dp), t)
    return dp[0][n-1], t

#arr = [4, 10, 3, 12, 20, 7]
arr = [4, 10, 20, 3, 12, 20, 7]
size = len(arr) - 1

min_cal, t = MatChainMul(arr, size)

print("Minimum number of multiplications are " + str(min_cal))
print(t)
print_parens(t, 1, size)


'''def matrix_chain_order(p):
    """
    Matrix-Chain-Order given a list of integers corresponding to the dimensions
    of each pair of matrices forming a chain.
    :param list p: A list of integers.
    
    >>> M, S = matrix_chain_order([2, 20, 4, 6])
    >>> print(M)
    {(1, 1): 0, (2, 2): 0, (3, 3): 0, (1, 2): 160, (2, 3): 480, (1, 3): 208}
    >>> print(S)
    {(1, 2): 1, (2, 3): 2, (1, 3): 2}
    """
    s = {}
    m = {}
    n = len(p)
    
    for i in range(1, n):
        m[tuple([i, i])] = 0
    
    for l in range(2, n):
        for i in range(1, n - l + 1):
            j = i + l - 1
            m[tuple([i, j])] = float('inf')
            for k in range(i, j):
                q = m[tuple([i, k])] + m[tuple([k + 1, j])] + (p[i-1] * p[k] * p[j])
                if q < m[tuple([i, j])]:
                    m[tuple([i, j])] = q
                    s[tuple([i, j])] = k
    return m, s


def print_optimal_parens(s, i, j):
    """
    Print the optimal parentheses according to the S-matrix computed by the
    matrix_chain_order function.
    :param dict s: A dictionary of tuples corresponding to the minimum k
                   values from each step of ``matrix_chain_order``.
    :param int i: Starting index.
    :param int j: End index.
    Example (continued from previous function):
    >>> M, S = matrix_chain_order([2, 20, 4, 6])
    >>> print_optimal_parens(S, 1, 3)
    ((A_1A_2)A_3)
    General form:
    
    >>> chain = [2, 20, 4, 6]
    >>> M, S = matrix_chain_order(chain)
    >>> print_optimal_parens(S, 1, len(S) - 1)
    ((A_1A_2)A_3)
    """

    if i == j:
        print("A_{}".format(i), end='')
    else:
        print('(', end='')
        print_optimal_parens(s, i, s[tuple([i, j])])
        print_optimal_parens(s, s[tuple([i, j])] + 1, j)
        print(')', end='')

        
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--chain',
                        default='4,10,3,12,20,7',
                        help="Specify comma-separated integers for input. [Default: 30,35,15,5,10,20,25]")
    parser.add_argument('-v',
                        '--verbose',
                        action='store_false',
                        help='Show the values of the S matrix.')

    args = parser.parse_args()
    chain = [int(i) for i in args.chain.split(',')]
    m, s = matrix_chain_order(chain)

    if args.verbose:
        for i,j in m:
            print('(i,j) = ({0},{1}): {2}'.format(i, j, m[tuple([i,j])]))

    print(s)
    
    print_optimal_parens(s, 1, len(chain) - 1)
    print()
'''

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