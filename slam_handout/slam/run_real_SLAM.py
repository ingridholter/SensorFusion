# %% Imports
from scipy.io import loadmat
from scipy.stats import chi2
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError as e:
    print(e)
    print("install tqdm for progress bar")

    # def tqdm as dummy
    def tqdm(*args, **kwargs):
        return args[0]


import numpy as np
from EKFSLAM import EKFSLAM
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from plotting import ellipse
from vp_utils import detectTrees, odometry, Car
from utils import rotmat2d

# %% plot config check and style setup


# to see your plot config
print(f"matplotlib backend: {matplotlib.get_backend()}")
print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

# try to set separate window ploting
if "inline" in matplotlib.get_backend():
    print("Plotting is set to inline at the moment:", end=" ")

    if "ipykernel" in matplotlib.get_backend():
        print("backend is ipykernel (IPython?)")
        print("Trying to set backend to separate window:", end=" ")
        import IPython

        IPython.get_ipython().run_line_magic("matplotlib", "")
    else:
        print("unknown inline backend")

print("continuing with this plotting backend", end="\n\n\n")


# set styles
try:
    # installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
    # gives quite nice plots
    plt_styles = ["science", "grid", "bright", "no-latex"]
    plt.style.use(plt_styles)
    print(f"pyplot using style set {plt_styles}")
except Exception as e:
    print(e)
    print("setting grid and only grid and legend manually")
    plt.rcParams.update(
        {
            # setgrid
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "k",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.5,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": True,
            "legend.numpoints": 1,
        }
    )


def main():
    # %% Load data
    victoria_park_foler = Path(
        __file__).parents[1].joinpath("data/victoria_park")
    realSLAM_ws = {
        **loadmat(str(victoria_park_foler.joinpath("aa3_dr"))),
        **loadmat(str(victoria_park_foler.joinpath("aa3_lsr2"))),
        **loadmat(str(victoria_park_foler.joinpath("aa3_gpsx"))),
    }

    timeOdo = (realSLAM_ws["time"] / 1000).ravel()
    timeLsr = (realSLAM_ws["TLsr"] / 1000).ravel()
    timeGps = (realSLAM_ws["timeGps"] / 1000).ravel()

    steering = realSLAM_ws["steering"].ravel()
    speed = realSLAM_ws["speed"].ravel()
    LASER = (
        realSLAM_ws["LASER"] / 100
    )  # Divide by 100 to be compatible with Python implementation of detectTrees
    La_m = realSLAM_ws["La_m"].ravel()
    Lo_m = realSLAM_ws["Lo_m"].ravel()

    K = timeOdo.size
    mK = timeLsr.size
    Kgps = timeGps.size

    # %% Parameters

    L = 2.83  # axel distance
    H = 0.76  # center to wheel encoder
    a = 0.95  # laser distance in front of first axel
    b = 0.5  # laser distance to the left of center

    car = Car(L, H, a, b)
    
    sigmas = np.array([2.5e-6, 1.25e-6, 0.15 * np.pi / 180])  # TODO tune, 2.5e-6, 1.25e-6, 0.15 deg
    CorrCoeff = np.array([[1, 0, 0], [0, 1, 0.9], [0, 0.9, 1]])
    Q = np.diag(sigmas) @ CorrCoeff @ np.diag(sigmas)
    # Q = CorrCoeff
    R = np.diag([0.1, 1 * np.pi / 180]) ** 2  # TODO tune, 0.1, 1 deg

    # first is for joint compatibility, second is individual
    JCBBalphas = np.array([1e-5, 1e-6])  # TODO tune, 1e-5, 1e-6

    sensorOffset = np.array([car.a + car.L, car.b])
    doAsso = True

    slam = EKFSLAM(Q, R, do_asso=doAsso, alphas=JCBBalphas,
                   sensor_offset=sensorOffset)

    # For consistency testing
    alpha = 0.05
    confidence_prob = 1 - alpha

    xupd = np.zeros((mK, 3))
    a = [None] * mK
    NIS = np.zeros(mK)
    NISnorm = np.zeros(mK)
    CI = np.zeros((mK, 2))
    CInorm = np.zeros((mK, 2))

    # Initialize state
    # you might want to tweak these for a good reference
    eta = np.array([Lo_m[0], La_m[1], 36 * np.pi / 180])
    P = np.zeros((3, 3))

    mk_first = 1  # first seems to be a bit off in timing
    mk = mk_first
    t = timeOdo[0]

    # %%  run
    N = 6000  # K

    doPlot = False

    lh_pose = None

    if doPlot:
        fig, ax = plt.subplots(num=1, clear=True)

        lh_pose = ax.plot(eta[0], eta[1], "k", lw=3)[0]
        sh_lmk = ax.scatter(np.nan, np.nan, c="r", marker="x")
        sh_Z = ax.scatter(np.nan, np.nan, c="b", marker=".")

    do_raw_prediction = True
    if do_raw_prediction:
        odos = np.zeros((K, 3))
        odox = np.zeros((K, 3))
        odox[0] = eta
        P_odo = P.copy()
        for k in range(min(N, K - 1)):
            odos[k + 1] = odometry(speed[k + 1], steering[k + 1], 0.025, car)
            odox[k + 1], _ = slam.predict(odox[k], P_odo, odos[k + 1])
            
    tot_num_asso = 0
    kgps = 1
    err_GPS = np.zeros([len(timeGps - 1),1])
    for k in tqdm(range(N)):
        if mk < mK - 1 and timeLsr[mk] <= timeOdo[k + 1]:
            # Force P to symmetric: there are issues with long runs (>10000 steps)
            # seem like the prediction might be introducing some minor asymetries,
            # so best to force P symetric before update (where chol etc. is used).
            # TODO: remove this for short debug runs in order to see if there are small errors
            P = (P + P.T) / 2
            dt = timeLsr[mk] - t
            if dt < 0:  # avoid assertions as they can be optimized avay?
                raise ValueError("negative time increment")

            # ? reset time to this laser time for next post predict
            t = timeLsr[mk]
            odo = odometry(speed[k + 1], steering[k + 1], dt, car)
            eta, P = slam.predict(eta, P, odo)  # TODO predict

            z = detectTrees(LASER[mk])
            eta, P, NIS[mk], a[mk] = slam.update(eta, P, z)  # TODO update

            num_asso = np.count_nonzero(a[mk] > -1)
            tot_num_asso += num_asso

            if num_asso > 0:
                NISnorm[mk] = NIS[mk] / (2 * num_asso)
                CInorm[mk] = np.array(chi2.interval(confidence_prob, 2 * num_asso)) / (
                    2 * num_asso
                )
            else:
                NISnorm[mk] = 1
                CInorm[mk].fill(1)

            xupd[mk] = eta[:3]
            
            # GPS Error Calculation
            t_next = timeLsr[mk + 1]
            t_prev = timeLsr[mk - 1]
            t_GPS = timeGps[kgps]
            
            dt_GPS = t_GPS - t
            dt_GPS_next = t_next - t_GPS
            dt_GPS_prev = t_GPS - t_prev
            if abs(dt_GPS_prev) < abs(dt_GPS) :
                err_GPS[kgps - 2] = np.linalg.norm(xupd[mk - 1, :2] - np.array([Lo_m[kgps], La_m[kgps]]))
                kgps += 1
                t_GPS = timeGps[kgps]
            
                dt_GPS = t_GPS - t
                dt_GPS_next = t_next - t_GPS
            if abs(dt_GPS_next) > abs(dt_GPS):              
                err_GPS[kgps - 1] = np.linalg.norm(xupd[mk, :2] - np.array([Lo_m[kgps], La_m[kgps]]))
                print(f"GPS TIME: {t_GPS}")
                print(f"LASER TIME: {t}")
                kgps += 1

            if doPlot:
                sh_lmk.set_offsets(eta[3:].reshape(-1, 2))
                if len(z) > 0:
                    zinmap = (
                        rotmat2d(eta[2])
                        @ (
                            z[:, 0] *
                            np.array([np.cos(z[:, 1]), np.sin(z[:, 1])])
                            + slam.sensor_offset[:, None]
                        )
                        + eta[0:2, None]
                    )
                    sh_Z.set_offsets(zinmap.T)
                lh_pose.set_data(*xupd[mk_first:mk, :2].T)

                ax.set(
                    xlim=[-200, 200],
                    ylim=[-200, 200],
                    title=f"step {k}, laser scan {mk}, landmarks {len(eta[3:])//2},\nmeasurements {z.shape[0]}, num new = {np.sum(a[mk] == -1)}",
                )
                plt.draw()
                plt.pause(0.00001)

            mk += 1

        if k < K - 1:
            dt = timeOdo[k + 1] - t
            t = timeOdo[k + 1]
            odo = odometry(speed[k + 1], steering[k + 1], dt, car)
            eta, P = slam.predict(eta, P, odo)

    # %% Consistency

    # NIS
    insideCI = (CInorm[:mk, 0] <= NISnorm[:mk]) * \
        (NISnorm[:mk] <= CInorm[:mk, 1])

    fig3, ax3 = plt.subplots(num=3, clear=True)
    ax3.plot(CInorm[:mk, 0], "--")
    ax3.plot(CInorm[:mk, 1], "--")
    ax3.plot(NISnorm[:mk], lw=0.5)

    ax3.set_title(f"NIS, {insideCI.mean()*100:.2f}% inside CI")
    ax3.set_xlabel('t [s]')
    fig3.savefig("NIS_real.pdf")
    
    CI_ANIS = np.array(chi2.interval(1 - alpha, 2*tot_num_asso)) / tot_num_asso
    ANIS = np.sum(NIS)/ tot_num_asso
    print(f"CI ANIS: {CI_ANIS}")
    print(f"ANIS: {ANIS}")

    # %% slam

    if do_raw_prediction:
        fig5, ax5 = plt.subplots(num=5, clear=True)
        ax5.scatter(
            Lo_m[timeGps < timeOdo[N - 1]],
            La_m[timeGps < timeOdo[N - 1]],
            c="r",
            marker=".",
            label="GPS",
        )
        ax5.plot(*odox[:N, :2].T, label="Odometry")
        ax5.plot(*xupd[mk_first:mk, :2].T, label="Estimate")
        ax5.grid()
        ax5.set_title("GPS vs odometry integration")
        ax5.set_xlabel('longitude [m]')
        ax5.set_ylabel('latitude [m]')
        ax5.legend()
        fig5.savefig("results_real.pdf")

    # %%
    fig6, ax6 = plt.subplots(num=6, clear=True)
    ax6.scatter(*eta[3:].reshape(-1, 2).T, color="r", marker="x", label="Landmark est.")
    ax6.plot(*xupd[mk_first:mk, :2].T, label="Pose est.")
    ax6.set(
        title=f"Steps {k}, laser scans {mk-1}, landmarks {len(eta[3:])//2},\nmeasurements {z.shape[0]}, num new = {np.sum(a[mk] == -1)}"
    )
    ax6.set_xlabel('longitude [m]')
    ax6.set_ylabel('latitude [m]')
    ax6.legend()
    fig6.savefig("Estimates_real.pdf")
    plt.show()


if __name__ == "__main__":
    main()
