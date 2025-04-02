import numpy as np  # All computations
from scipy.stats import norm  # To compute the CDF of Gaussian distribution in the PMF computation
import matplotlib.pyplot as plt  # Plotting
from tqdm import tqdm  # To view progess bar


def compute_entropy(P):
    """
    Compute the entropy H of a discrete PMF P
    """
    P_supp = P[P != 0]  # Take only where P is supported
    H = -np.sum(P_supp * np.log(P_supp))
    return H


def compute_NMI(P):
    """
    Computes the NMI of a joint PMF P of (X, Y) over the [d] x [d] discrete space
    """
    P_X = np.sum(P, axis=0)
    P_Y = np.sum(P, axis=1)
    H_X = compute_entropy(P_X)
    H_Y = compute_entropy(P_Y)
    H_XY = compute_entropy(P)
    NMI = (H_X + H_Y) / H_XY
    return NMI


def compute_empirical_joint_distribution(X, Y):
    """
    Computes the empirical joint PMF of (X, Y) over the [d] x [d] discrete space
    """
    N_p = np.size(X)
    P_XY = np.zeros((256, 256))

    for i in range(N_p):
        P_XY[X[i], Y[i]] += 1

    assert N_p == np.sum(P_XY), "mismatch in counts"
    return P_XY / N_p
        

def compute_empirical_joint_posterior_distribution(X, Y, vec_PMF_sigma_X, PMF_sigma_Y):

    N_p = np.size(X)
    smear_Y = np.size(PMF_sigma_Y)
    P_XY = np.zeros((256, 256))

    for i in range(N_p):

        PMF_sigma_X = vec_PMF_sigma_X[i]
        smear_X = np.size(PMF_sigma_X)
        P_XY[X[i], Y[i]] += PMF_sigma_X[0] * PMF_sigma_Y[0]

        for k1 in range(1, smear_X):

            for k2 in range(1, smear_Y):

                P_XY[np.clip(X[i] + k1, 0, 255), np.clip(Y[i] + k2, 0, 255)] += PMF_sigma_X[k1] * PMF_sigma_Y[k2]
                P_XY[np.clip(X[i] + k1, 0, 255), np.clip(Y[i] - k2, 0, 255)] += PMF_sigma_X[k1] * PMF_sigma_Y[k2]
                P_XY[np.clip(X[i] - k1, 0, 255), np.clip(Y[i] + k2, 0, 255)] += PMF_sigma_X[k1] * PMF_sigma_Y[k2]
                P_XY[np.clip(X[i] - k1, 0, 255), np.clip(Y[i] - k2, 0, 255)] += PMF_sigma_X[k1] * PMF_sigma_Y[k2]

    return P_XY / np.sum(P_XY)


# Dimensions of the problem
N_w = 6 
N_d = 11
N_p = N_w * N_d
sigma = 5
mu = 128

# Initialize camera/mount parameters
f = 0.0367
h = 60
theta = 36 
theta_rad = theta * np.pi/180
s = 20

# Compute the transformed tile areas for the above mounted camera
A = np.zeros([N_w, N_d])

for j in range(N_d):

    A[:, j] = f**2 * h * s / (2 * np.cos(theta_rad)) \
                * ( 1/((j+1) * s * np.cos(theta_rad) + h * np.sin(theta_rad))**2 \
                   - 1/((j+2) * s * np.cos(theta_rad) + h * np.sin(theta_rad))**2 )

vec_A = np.ravel(A)


print("Area boost in SSNR = ", -10*np.log10(vec_A))

SINR_dB = 10
SINR = 10**(SINR_dB/10)
sigma_I = sigma / np.sqrt(SINR)

# Create Discretized Gaussian PMF (2*smear_width-1, 2*smear_width-1) indexed as | -(smear_width-1) | ... | -2 | -1 | 0 | 1 | ... | (smear_width-1) |
smear_width = 3

# Define only for non-negative indices (center + right half) as the PMF is symmetric
PMF_I = np.zeros(smear_width)

# Discretized Gaussian kernel N(0, sigma_I) of size (2*smear_width-1, 2*smear_width-1)
for k in range(len(PMF_I)):

    PMF_I[k] = norm.cdf((k+0.5), scale=sigma_I)-norm.cdf((k-0.5), scale=sigma_I)

PMF_I /= (PMF_I[0] + 2 * np.sum(PMF_I[1:])) # Normalize to be a valid PMF

# Create a dirac PMF (2*smear_width-1, 2*smear_width-1) with similar indexing with values | 0 | ... | 0 | 1 | 0 | ... | 0 |
PMF_I_dirac = np.zeros(smear_width)
PMF_I_dirac[smear_width // 2] = 1

# Initialize
N_s = 50  # Number of statistical samples

SSNR_dB = np.arange(80, 30, -5)
SSNR = 10**(SSNR_dB/10)
N_0 = sigma**2 / SSNR
n_SNR_values = np.size(SSNR_dB)

Pe_NMI = np.zeros(n_SNR_values)
Pe_ENMI1D = np.zeros(n_SNR_values)
Pe_ENMI2D = np.zeros(n_SNR_values)

for i in range(n_SNR_values):

    vec_sigma_S = np.sqrt(N_0[i] / vec_A)
    vec_PMF_S_2D = []
    vec_PMF_S_1D = []

    for n in range(N_p):

        sigma_S_1D = vec_sigma_S[n]
        sigma_S_2D = np.sqrt((sigma_I**2+sigma_S_1D**2))
        PMF_S_1D = np.zeros(smear_width)
        PMF_S_2D = np.zeros(smear_width)

        for k in range(smear_width):
            PMF_S_2D[k] = norm.cdf((k+0.5), scale=sigma_S_2D)-norm.cdf((k-0.5), scale=sigma_S_2D)
            PMF_S_1D[k] = norm.cdf((k+0.5), scale=sigma_S_1D)-norm.cdf((k-0.5), scale=sigma_S_1D)

        PMF_S_1D /= (PMF_S_1D[0] + 2 * np.sum(PMF_S_1D[1:]))
        PMF_S_2D /= (PMF_S_2D[0] + 2 * np.sum(PMF_S_2D[1:]))
        vec_PMF_S_2D.append(PMF_S_2D)
        vec_PMF_S_1D.append(PMF_S_1D)

    Ne_NMI = 0
    Ne_ENMI1D = 0
    Ne_ENMI2D = 0

    for _ in tqdm(range(N_s)):

        signal_0 = mu + sigma * np.random.randn(N_p)
        signal_1 = mu + sigma * np.random.randn(N_p)

        z_S = vec_sigma_S * np.random.randn(N_p)
        z_I = sigma_I * np.random.randn(N_p)

        z_I_0 = sigma_I * np.random.randn(N_p)
        z_I_1 = sigma_I * np.random.randn(N_p)

        captured_signal = np.clip(signal_0 + z_S + z_I, 0, 255).astype(np.uint8)
        map_0 = np.clip(signal_0 + z_I_0, 0, 255).astype(np.uint8)
        map_1 = np.clip(signal_1 + z_I_1, 0, 255).astype(np.uint8)

        joint_ENMI_2D_0 = compute_empirical_joint_posterior_distribution(captured_signal, map_0, vec_PMF_S_2D, PMF_I)
        joint_ENMI_2D_1 = compute_empirical_joint_posterior_distribution(captured_signal, map_1, vec_PMF_S_2D, PMF_I)

        joint_ENMI_1D_0 = compute_empirical_joint_posterior_distribution(captured_signal, map_0, vec_PMF_S_1D, PMF_I_dirac)
        joint_ENMI_1D_1 = compute_empirical_joint_posterior_distribution(captured_signal, map_1, vec_PMF_S_1D, PMF_I_dirac)

        joint_NMI_0 = compute_empirical_joint_distribution(captured_signal, map_0)
        joint_NMI_1 = compute_empirical_joint_distribution(captured_signal, map_1)

        ENMI2D_0 = compute_NMI(joint_ENMI_2D_0)
        ENMI2D_1 = compute_NMI(joint_ENMI_2D_1)

        ENMI1D_0 = compute_NMI(joint_ENMI_1D_0)
        ENMI1D_1 = compute_NMI(joint_ENMI_1D_1)

        NMI_0 = compute_NMI(joint_NMI_0)
        NMI_1 = compute_NMI(joint_NMI_1)

        Ne_NMI += (NMI_0 < NMI_1)
        Ne_ENMI1D += (ENMI1D_0 < ENMI1D_1)
        Ne_ENMI2D += (ENMI2D_0 < ENMI2D_1)

    Pe_NMI[i] = Ne_NMI / N_s
    Pe_ENMI1D[i] = Ne_ENMI1D / N_s
    Pe_ENMI2D[i] = Ne_ENMI2D / N_s

    print(f"SSNR = {SSNR_dB[i]}dB | NMI = {Pe_NMI[i]} | ENMI1D = {Pe_ENMI1D[i]} | ENMI2D = {Pe_ENMI2D[i]} ")

plt.semilogx(N_0, Pe_NMI, label=r"$NMI$", ls=':', color='k')
plt.semilogx(N_0, Pe_ENMI1D, label=r"$ENMI_{1D}$", ls='--', color='k')
plt.semilogx(N_0, Pe_ENMI2D, label=r"$ENMI_{2D}$", color='k')
plt.grid()
plt.legend()
plt.xlabel("N0")
plt.ylabel("Pe")
plt.title(f"SINR = {SINR_dB}dB")
plt.savefig('MI.png', dpi=400, bbox_inches='tight')

# Save the numpy arrays
np.save("N_0_MI.npy", N_0)
np.save("Pe_NMI.npy", Pe_NMI)
np.save("Pe_ENMI1D.npy", Pe_ENMI1D)
np.save("Pe_ENMI2D.npy", Pe_ENMI2D)
