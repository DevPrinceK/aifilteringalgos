
import random
import numpy as np

from models import *


#
# Add your Filtering / Smoothing approach(es) here
#
class FilterSmoother:
    def __init__(self, probs, tm, om, sm):
        self.__tm = tm
        self.__om = om
        self.__sm = sm
        self.__current_f = probs # initialising with dummy/step0-values
        self.__current_fb = probs # initialising with dummy/step0-values
        
        
    # sensorR is the sensor reading (index!) in step t_plus_one, f_t is the probability distribution in step t
    #
    # self.__current_f is the probability distribution resulting from the filtering    
    def filter(self, sensorR : int, f_t : np.array) -> np.array :        #print( self.__f)
        self.__current_f = f_t
 
        # add your code here 

        return self.__current_f

    # sensor_r_seq is the sequence (array) with the t-k sensor readings for smoothing, 
    # f_k is the filtered result (f_vector) for step k
    # OBS: f_k is not necessarily the same as self.__current_f, but it *can* be; that depends on how you handle the control
    # loop(s) for filtering and smoothing. The assumption made by Elin is that the control loop over t is *outside* the
    # calculations / methods, while the inner loop from t to t-k is inside the smoothing.
    # 
    # self.__current_fb is the smoothed result (fb_vector)
    def smooth(self, sensor_r_seq : np.array, f_k : np.array) -> np.array:
        self.__current_fb = f_k # in case there is no window to smooth over, just return the filtered result

    	# add you code here        
        return self.__current_fb
