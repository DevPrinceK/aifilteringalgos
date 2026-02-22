
#
# The Localizer binds the models together and controls the update cycle in its "update" method.
# That method is called from the visualisation (Dashboard) and calls in turn the RobotSim methods as well
# as "filter(...)" in "Filters.HMMFilter"
#

import numpy as np
import matplotlib.pyplot as plt
import random

from models import *
import Filters

class Localizer:
    def __init__(self, sm, uniformF, t_minus_k):

        self.__sm = sm

        self.__tm = TransitionModel(self.__sm)
        if uniformF == True:
            self.__om = ObservationModel_UF.ObservationModelUF(self.__sm)
        else: 
            self.__om = ObservationModel_NUF.ObservationModel(self.__sm)

        self.__t_minus_k = t_minus_k
        # change in initialise in case you want to start out with something else...
        self.initialise()

    # retrieve the transition model that we are currently working with
    def get_transition_model(self) -> np.array:
        return self.__tm

    # retrieve the observation model that we are currently working with
    def get_observation_model(self) -> np.array:
        return self.__om

    # the current true pose (x, h, h) that should be kept in the local variable __trueState
    def get_current_true_pose(self) -> (int, int, int):
        x, y, h = self.__sm.state_to_pose(self.__trueState_seq[0])
        return x, y, h

    # the current probability distribution over all states (fb = f if not smoothing)
    def get_current_f_vector(self) -> np.array(float):
        return self.__fb_probs

    # the current sensor reading (as position in the grid). "Nothing" is expressed as None
    def get_current_reading(self) -> (int, int):
        ret = None
        if self.__sense != None:
            ret = self.__sm.reading_to_position(self.__sensor_seq[0])
        return ret;

    # get the currently most likely position, based on single most probable pose
    def most_likely_position(self) -> (int, int):
        return self.__estimate

    # (re-)initialise for a new run without change of size
    def initialise(self):
        self.__trueState_seq = []
        self.__trueState_seq.append(random.randint(0, self.__sm.get_num_of_states() - 1))
        self.__sensor_seq = []
        self.__sensor_seq.append(None)
        
        self.__f_probs = np.ones(self.__sm.get_num_of_states()) / (self.__sm.get_num_of_states())
        self.__fb_probs = self.__f_probs
        self.__estimate = self.__sm.state_to_position(np.argmax(self.__f_probs))

        # simple evaluation measures that go into the visualisation, add more if you like!
        
        self.__rs = RobotSim( self.__trueState_seq[0], self.__sm)
        self.__HMM = Filters.FilterSmoother( self.__f_probs, self.__tm, self.__om, self.__sm)
 
        # initialising sequences for smoothing
        for i in range(self.__t_minus_k) :
            self.__trueState_seq.append(self.__rs.move_once(self.__tm))
            self.__sensor_seq.append(self.__rs.sense_in_current_state(self.__om))


    #
    #  The update cycle:
    #  - robot moves one step, generates new state / pose
    #  - sensor produces one reading based on the true state / pose
    #  - filtering approach produces new probability distribution based on
    #  sensor reading, transition and sensor models
    #
    #  Reports back to the caller (viewer):
    #  Return
    #  - true if sensor reading was not "nothing", else false,
    #  - AND the three values for the (new) true pose (x, y, h),
    #  - AND the two values for the (current) sensor reading (if not "nothing")
    #  - AND the error made in this step
    #  - AND the new probability distribution
    #
    def update(self) -> (bool, int, int, int, int, int, int, int, int, np.array(1)) :
        self.__trueState_seq.append( self.__rs.move_once(self.__tm))
        self.__sensor_seq.append( self.__rs.sense_in_current_state(self.__om))

        sense = self.__sensor_seq.pop(0)
        trueState = self.__trueState_seq.pop(0)

        
        self.__f_probs = self.__HMM.filter( sense, self.__f_probs)

        # returns f_probs if no smoothing, but making it explicit is better
        self.__fb_probs = self.__HMM.smooth( self.__sensor_seq, self.__f_probs)

        
        fPositions = self.__fb_probs.copy()
        
        for state in range(0, self.__sm.get_num_of_states(), 4) :
            fPositions[state:state+4] = sum(fPositions[state:state+4])
            
        self.__estimate = self.__sm.state_to_position(np.argmax(fPositions))
        
        ret = False  # in case the sensor reading is "nothing" this is kept...
        tsX, tsY, tsH = self.__sm.state_to_pose(trueState)
        srX = -1
        srY = -1
        if sense != None:
            srX, srY = self.__sm.reading_to_position(sense)
            ret = True
            
        eX, eY = self.__estimate
       
        error = abs(tsX-eX)+abs(tsY-eY)                
                       
        
        return ret, tsX, tsY, tsH, srX, srY, eX, eY, error, fPositions

    # update in case the true pose (trajectory) is known and fed into the Localizer from an external source
    def updateWTruePose(self, trueState) -> (bool, int, int, int, int, int, int, int, int, np.array(1)) :
        self.__trueState_seq[0] = trueState
        self.__sensor_seq[0] = self.__rs.sense_in_current_state()
        self.__probs = self.__HMM.filter( self.__sensor_seq[0])
        
        fPositions = self.__probs.copy()
        
        for state in range(0, self.__sm.get_num_of_states(), 4) :
            fPositions[state:state+4] = sum(fPositions[state:state+4])
            
      
        self.__estimate = self.__sm.state_to_position(np.argmax(fPositions))
        
        ret = False  # in case the sensor reading is "nothing" this is kept...
        tsX, tsY, tsH = self.__sm.state_to_pose(self.__trueState_seq[0])
        srX = -1
        srY = -1
        if self.__sensor_seq[0] != None:
            srX, srY = self.__sm.reading_to_position(self.__sensor_seq[0])
            ret = True
            
        eX, eY = self.__estimate
       
        error = abs(tsX-eX)+abs(tsY-eY)                
                       
        
        return ret, tsX, tsY, tsH, srX, srY, eX, eY, error, fPositions
