import time
import asyncio

def main():
    i = 0
    while True:
        time.sleep(1)
        yield i
        i += 1



t = time.perf_counter()
g = main()
print(f'{time.perf_counter()-t}')
for _ in g:
    print(f'{time.perf_counter()-t} : {_}')