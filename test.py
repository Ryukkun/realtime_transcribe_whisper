from concurrent.futures import ThreadPoolExecutor
import time

def wawa():
    time.sleep(1)
    print('fin')


exe = ThreadPoolExecutor(1)
exe.submit(wawa)
print('submit fin')
time.sleep(2)