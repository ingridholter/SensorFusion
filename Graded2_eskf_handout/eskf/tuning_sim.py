import numpy as np
from quaternion import RotationQuaterion
from datatypes.eskf_params import ESKFTuningParams
from datatypes.eskf_states import NominalState, ErrorStateGauss

tuning_params_sim = ESKFTuningParams(
    accm_std= 10 * 0.07 / np.sqrt(3600), #0.0012, velocity random walk 
    accm_bias_std= 500* 0.007 * 9.81 / 1000, #0.000069, acceleration random walk
    accm_bias_p= 0.01 * 1.89 / 150, #0.0126

    gyro_std= 1000 * 0.15 * np.pi / (180 * np.sqrt(3600)), #0.000044, angle random walk
    gyro_bias_std= 100 * np.sqrt(0.09 * np.pi / (180 * 3600)), #0.00066, rate random walk
    gyro_bias_p= 0.1 * 1.89 / 800, #0.0024

    gnss_std_ne=0.5,
    gnss_std_d=2.)

x_nom_init_sim = NominalState(
    np.array([0., 0., 0.]),  # position
    np.array([0., 0., 0.]),  # velocity
    RotationQuaterion.from_euler([0., 0., 0.]),  # orientation
    np.zeros(3),  # accelerometer bias
    np.zeros(3),  # gyro bias
    ts=0.)

init_std_sim = np.repeat(repeats=3,  # repeat each element 3 times
                         a=[1.,  # position
                            1.,  # velocity
                            np.deg2rad(0.1),  # angle vector
                            0.1,  # accelerometer bias
                            0.001])  # gyro bias
x_err_init_sim = ErrorStateGauss(np.zeros(15), np.diag(init_std_sim**2), 0.)
