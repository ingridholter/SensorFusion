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
    solution.used["nis_nees.get_NIS"] = True
    """Calculate NIS

    Args:
        z_gnss (GnssMeasurement): gnss measurement
        z_gnss_pred_gauss (MultiVarGaussStamped): predicted gnss measurement
        marginal_idxs (Optional[Sequence[int]]): Sequence of marginal indexes.
            For example used for calculating NIS in only xy direction.  

    Returns:
        NIS (float): NIS value
    """
    if marginal_idxs:
        pred = z_gnss_pred_gauss.marginalize(marginal_idxs)
        z = z_gnss.pos[marginal_idxs]
    else:
        pred = z_gnss_pred_gauss
        z = z_gnss.pos
    NIS = pred.mahalanobis_distance_sq(z)

    return NIS


def get_error(x_true: NominalState,
              x_nom: NominalState,
              ) -> 'ndarray[15]':
    solution.used["nis_nees.get_error"] = True
    """Finds the error (difference) between True state and 
    nominal state. See (Table 10.1).


    Returns:
        error (ndarray[15]): difference between x_true and x_nom. 
    """
    pos_diff = x_true.pos - x_nom.pos
    vel_diff = x_true.vel - x_nom.vel

    ori_diff = (x_nom.ori.conjugate()@(x_true.ori)).as_avec()

    accm_bias_diff = x_true.accm_bias - x_nom.accm_bias
    gyro_bias_diff = x_true.gyro_bias - x_nom.gyro_bias

    error = np.concatenate(
        [pos_diff, vel_diff, ori_diff, accm_bias_diff, gyro_bias_diff])

    return error


def get_NEES(error: 'ndarray[15]',
             x_err: ErrorStateGauss,
             marginal_idxs: Optional[Sequence[int]] = None
             ) -> float:
    solution.used["nis_nees.get_NEES"] = True
    """Calculate NEES

    Args:
        error (ndarray[15]): errors between x_true and x_nom (from get_error)
        x_err (ErrorStateGauss): estimated error
        marginal_idxs (Optional[Sequence[int]]): Sequence of marginal indexes.
            For example used for calculating NEES for only the position. 

    Returns:
        NEES (float): NEES value
    """
    if marginal_idxs:
        err_est = x_err.marginalize(marginal_idxs)
        err_true = error[marginal_idxs]
    else:
        err_est = x_err
        err_true = error
    NEES = err_est.mahalanobis_distance_sq(err_true)
    return NEES


def get_time_pairs(unique_data, data):
    """match data from two different time series based on timestamps"""
    gt_dict = dict(([x.ts, x] for x in unique_data))
    pairs = [(gt_dict[x.ts], x) for x in data if x.ts in gt_dict]
    times = [pair[0].ts for pair in pairs]
    return times, pairs
