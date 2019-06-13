import multiprocessing

NUMS = list(range(1, 40))


def fib(n):
    if n <= 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)


if __name__ == '__main__':
    import time

    start = time.time()
    results = []
    for num in NUMS:
        results.append(fib(num))
    end = time.time()
    print(f"duration {end-start}")
    print(results)


    with multiprocessing.Pool(8) as pool:
        start = time.time()
        results = pool.map(fib, NUMS)
        end = time.time()
        print(f"duration {end-start}")
        print(results)

        # start = time.time()
        # results = pool.starmap(fib, NUMS)
        # end = time.time()
        # print(f"duration {end-start}")
        # print(results)
