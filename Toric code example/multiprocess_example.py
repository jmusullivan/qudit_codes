import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Process, Pool
import time

from TCd_decoder import *




code = TCd(10,10)
code.generate_code() 
p=.3

def threshold(runs):
  logical_success = 0
  for run in range(runs):
    error = GF(Error(p,code.H))
    e_corr, syndrome_after_decoding = my_decoder(code.T, error)

    if (error - e_corr)@ GF(code.Z1) ==0 and (error - e_corr)@ GF(code.Z2) ==0:
        logical_success+=1

  return logical_success/runs


print('starting calculation')


if __name__ == '__main__':
  start_time = time.perf_counter()
  with Pool() as pool:
    result = pool.map(threshold, [100 for i in range(10)])
  finish_time = time.perf_counter()
  print(result)
  print("Program finished in {} seconds - using multiprocessing".format(finish_time-start_time))
#########################################
  start_time = time.perf_counter()
  result = []
  for x in [100 for i in range(10)]:
    result.append(threshold(x))
  finish_time = time.perf_counter()
  print(result)
  print("Program finished in {} seconds".format(finish_time-start_time))
  
  

