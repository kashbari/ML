## Timer Class for Timing functions

import time
import numpy as np

class Timer:
    """ Records multiple running times """
    def __init__(self):
        self.times = []
        #self.start() #Uncomment to start timer upon initialization

    def start(self):
        """ Starts Timer """
        self.tik = time.time()

    def stop(self):
        """ Stops Timer and records in list """
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def time_func(self,func,x):
        """ Times a function """
        self.start()
        y = func(x)
        self.stop()
        return y

    def average(self):
        """ Returns average time """
        return sum(self.times)/len(self.times)

    def sum(self):
        """ Return sum of times """
        return sum(self.times)

    def cumsum(self):
        """ Cumulative sum of times """
        return np.array(self.times).cumsum().tolist()

## To use

#t = Timer()
#t.start() #Starts timer
#f'{t.stop():.5f} sec } #Starts timer
