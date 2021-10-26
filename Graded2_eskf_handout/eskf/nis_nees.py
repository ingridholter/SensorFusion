import numpy as np
from numpy import ndarray
from typing import Sequence, Optional

from datatypes.measurements import GnssMeasurement
from datatypes.eskf_states import NominalState, ErrorStateGauss
from datatypes.multivargaussian import MultiVarGaussStamped

import solution


def get_NIS(z_gnss: GnssMeasurement,
            z_gnss_pred_gauss: MultiVarGaussStamped,
            marginal_idxs: Optional[Sequence[int]] = None
            ) -> float:
    """Calculate NIS

    Args:
        z_gnss (GnssMeasurement): gnss measurement
        z_gnss_pred_gauss (MultiVarGaussStamped): predicted gnss measurement
        marginal_idxs (Optional[Sequence[int]]): Sequence of marginal indexes.
            For example used for calculating NIS in only xy direction.  

    Returns:
        NIS (float): NIS value
    """
    pos = z_gnss.pos
    if marginal_idxs != None:
        z_gnss_pred_gauss = z_gnss_pred_gauss.marginalize(marginal_idxs)
        pos = pos[marginal_idxs]
        
    NIS = z_gnss_pred_gauss.mahalanobis_distance_sq(pos)
        
    # NIS_sol = solution.nis_nees.get_NIS(z_gnss, z_gnss_pred_gauss, marginal_idxs)

    return NIS


def get_error(x_true: NominalState,
              x_nom: NominalState,
              ) -> 'ndarray[15]':
    """Finds the error (difference) between True state and 
    nominal state. See (Table 10.1).
    

    Returns:
        error (ndarray[15]): difference between x_true and x_nom. 
    """
    error = np.zeros(15)
    error[:3] = x_true.pos - x_nom.pos
    error[3:6] = x_true.vel - x_nom.vel
    
    d_q = x_nom.ori.conjugate().multiply(x_true.ori)
    error[6:9] = d_q.as_avec()
    
    error[9:12] =  x_true.accm_bias - x_nom.accm_bias
    error[12:15] = x_true.gyro_bias - x_nom.gyro_bias
        
    # error_sol = solution.nis_nees.get_error(x_true, x_nom)

    return error


def get_NEES(error: 'ndarray[15]',
             x_err: ErrorStateGauss,
             marginal_idxs: Optional[Sequence[int]] = None
             ) -> float:
    """Calculate NEES

    Args:
        error (ndarray[15]): errors between x_true and x_nom (from get_error)
        x_err (ErrorStateGauss): estimated error
        marginal_idxs (Optional[Sequence[int]]): Sequence of marginal indexes.
            For example used for calculating NEES for only the position. 

    Returns:
        NEES (float): NEES value
    """
    if marginal_idxs != None:
        error = error[marginal_idxs]
        x_err = x_err.marginalize(marginal_idxs)
    NEES = x_err.mahalanobis_distance_sq(error)
       
    # NEES_sol = solution.nis_nees.get_NEES(error, x_err, marginal_idxs)

    return NEES


def get_time_pairs(unique_data, data):
    """match data from two different time series based on timestamps"""
    gt_dict = dict(([x.ts, x] for x in unique_data))
    pairs = [(gt_dict[x.ts], x) for x in data if x.ts in gt_dict]
    times = [pair[0].ts for pair in pairs]
    return times, pairs
