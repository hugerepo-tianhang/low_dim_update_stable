import multiprocessing

def f(x):
    print (multiprocessing.current_process()._identity)
    if multiprocessing.current_process()._identity == (1,):
        print("sss")
    return x * x

p = multiprocessing.Pool()
print (p.map(f, range(6)))