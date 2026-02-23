
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

        # Cache transition matrices once for speed.
        # TransitionModel stores T as T[i, j] = P(X_{t+1}=j | X_t=i).
        # With belief vectors represented as column vectors, prediction is:
        #   f_pred = T^T @ f_t
        self.__T = self.__tm.get_T()
        self.__T_transp = self.__tm.get_T_transp()
        self.__num_states = self.__sm.get_num_of_states()

        self.__current_f = probs # initialising with dummy/step0-values
        self.__current_fb = probs # initialising with dummy/step0-values

    def __o_vec(self, reading: int | None) -> np.ndarray:
        """Return the observation likelihood vector o where o[i] = P(r | X=i).

        We intentionally do NOT build the full diagonal matrix O_r (NxN), because that
        would be wasteful. The forward equations only need the diagonal as a vector.
        """
        # Using the public API for clarity/robustness (works for both UF and NUF models).
        return np.fromiter(
            (self.__om.get_o_reading_state(reading, i) for i in range(self.__num_states)),
            dtype=float,
            count=self.__num_states,
        )

    @staticmethod
    def __normalize(vec: np.ndarray) -> np.ndarray:
        """Normalize a probability vector; fall back safely if it is degenerate."""
        s = float(np.sum(vec))
        if s <= 0.0 or not np.isfinite(s):
            # Should not happen with the provided models, but guard against numerical issues.
            return np.ones_like(vec, dtype=float) / float(vec.size)
        return vec / s
        
        
    # sensorR is the sensor reading (index!) in step t_plus_one, f_t is the probability distribution in step t
    #
    # self.__current_f is the probability distribution resulting from the filtering    
    def filter(self, sensorR : int, f_t : np.array) -> np.array :        #print( self.__f)
        # Forward Filtering for an HMM:
        #   f_pred = T^T @ f_t
        #   f_{t+1} = alpha * O_{r_{t+1}} * f_pred
        # where alpha is the normalisation constant.

        f_t = np.asarray(f_t, dtype=float).reshape(-1)
        if f_t.size != self.__num_states:
            raise ValueError(
                f"Expected belief vector of length {self.__num_states}, got {f_t.size}."
            )

        # 1) Predict (motion update)
        f_pred = self.__T_transp @ f_t

        # 2) Update (sensor correction)
        o = self.__o_vec(sensorR)
        f_next_unnorm = o * f_pred

        # 3) Normalise
        self.__current_f = self.__normalize(f_next_unnorm)
        return self.__current_f

    # sensor_r_seq is the sequence (array) with the t-k sensor readings for smoothing, 
    # f_k is the filtered result (f_vector) for step k
    # OBS: f_k is not necessarily the same as self.__current_f, but it *can* be; that depends on how you handle the control
    # loop(s) for filtering and smoothing. The assumption made by Elin is that the control loop over t is *outside* the
    # calculations / methods, while the inner loop from t to t-k is inside the smoothing.
    # 
    # self.__current_fb is the smoothed result (fb_vector)
    def smooth(self, sensor_r_seq : np.array, f_k : np.array) -> np.array:
        # Fixed-lag Forward–Backward smoothing:
        # Given the filtered distribution f_k = P(X_k | e_1:k) and a short future
        # evidence sequence e_{k+1 : k+L}, compute
        #   beta_k(x_k) = P(e_{k+1 : k+L} | x_k)
        # and return
        #   fb_k(x_k) ∝ f_k(x_k) * beta_k(x_k)
        #
        # IMPORTANT:
        # The Localizer passes a rolling buffer `sensor_r_seq` that is typically of
        # length (L+1): the last element is kept for the next iteration. Therefore we
        # use only the first L = len(sensor_r_seq)-1 readings here.

        f_k = np.asarray(f_k, dtype=float).reshape(-1)
        if f_k.size != self.__num_states:
            raise ValueError(
                f"Expected belief vector of length {self.__num_states}, got {f_k.size}."
            )

        if sensor_r_seq is None:
            self.__current_fb = self.__normalize(f_k)
            return self.__current_fb

        readings = list(sensor_r_seq)
        lag = max(0, len(readings) - 1)
        if lag == 0:
            # No smoothing window -> return filtered belief.
            self.__current_fb = self.__normalize(f_k)
            return self.__current_fb

        # Backward message initialisation: beta_{k+lag} = 1 for all states.
        beta = np.ones(self.__num_states, dtype=float)

        # Incorporate evidence from k+1 up to k+lag.
        # Recurrence: beta_{t-1} = T @ (O_{e_t} * beta_t)
        for r in reversed(readings[:lag]):
            beta = self.__T @ (self.__o_vec(r) * beta)

        fb_unnorm = f_k * beta
        self.__current_fb = self.__normalize(fb_unnorm)
        return self.__current_fb
