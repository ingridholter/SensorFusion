import numpy as np
from quaternion import RotationQuaterion
from datatypes.eskf_params import ESKFTuningParams
from datatypes.eskf_states import NominalState, ErrorStateGauss

tuning_params_real = ESKFTuningParams(

    accm_std=0.0012,
    accm_bias_std=1.5e-5,  # 0.002
    accm_bias_p=1e-5,


    gyro_std=4.4e-5,
    gyro_bias_std=2e-4,  # 7e-6
    gyro_bias_p=1e-7,

    gnss_std_ne=0.35,
    gnss_std_d=1,

    use_gnss_accuracy=False)

x_nom_init_real = NominalState(
    np.array([0., 0., 0.]),  # position
    np.array([0., 0., 0.]),  # velocity
    RotationQuaterion.from_euler([0., 0., np.pi]),  # orientation
    np.zeros(3),  # accelerometer bias
    np.zeros(3),  # gyro bias
    ts=0.)

init_std_real = np.repeat(repeats=3,  # repeat each element 3 times
                          a=[1,  # position
                             10,  # velocity
                             np.deg2rad(10),  # angle vector
                             0.1,  # accelerometer bias
                             0.001])  # gyro bias

x_err_init_real = ErrorStateGauss(np.zeros(15), np.diag(init_std_real**2), 0.)
