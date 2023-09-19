# 재귀 방법
def fib(n):
    if n==1 or n==2:
        return 1
    else:
        return (fib(n-1)+fib(n-2))
    
def fibonacci(n):
    return n-2

n = int(input())
print(fib(n), fibonacci(n))


# DP 활용
def fib(n):
    f = [0] * (n+1)
    f[1] = f[2] = 1
    for i in range(3,n+1):
        f[i] = f[i-1]+f[i-2]
    return f[n]
    
def fibonacci(n):
    return n-2

n = int(input())
print(fib(n), fibonacci(n))
