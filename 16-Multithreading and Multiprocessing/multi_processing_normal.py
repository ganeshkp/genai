## PRocesses that run in parallel
### CPU-Bound Tasks-Tasks that are heavy on CPU usage (e.g., mathematical computations, data processing).
## PArallel execution- Multiple cores of the CPU

import multiprocessing

import time

def square_numbers():
    for i in range(5):
        time.sleep(1)
        print(f"Square: {i*i}")

def cube_numbers():
    for i in range(5):
        time.sleep(1.5)
        print(f"Cube: {i * i * i}")

if __name__=="__main__":
    start_time=time.time()

    square_numbers()
    cube_numbers()

    finished_time=time.time()-start_time
    print(finished_time)