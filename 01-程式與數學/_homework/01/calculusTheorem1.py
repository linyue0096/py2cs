def df(f, x):
    # ...

def integrate(f, a, b):
    # ...

def theorem1(f, x):
    assert df(lambda x:integrate(f, 0, x), x) == f(x)
