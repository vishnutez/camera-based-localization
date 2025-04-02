import numpy as np  # All computations
from scipy.stats import norm  # To compute the CDF of Gaussian distribution in the PMF computation
import matplotlib.pyplot as plt  # Plotting
from tqdm import tqdm  # To view progess bar


def compute_entropy(P):
    """
    Compute the entropy H of i discrete PMF P
    """
    P_supp = P[P != 0]  # Take only where P is supported
    H = -np.sum(P_supp * np.log(P_supp))
    return H


def compute_NMI(P):
    """
    Computes the NMI of i joint PMF P of (X, Y) over the [V] x [V] discrete space
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
    Computes the empirical joint PMF of (X, Y) over the [V] x [V] discrete space
    """
    N_p = np.size(X)
    V = 256
    P_XY = np.zeros((V, V))

    for t in range(N_p):
        P_XY[X[t], Y[t]] += 1

    assert N_p == np.sum(P_XY), "mismatch in counts"
    return P_XY / N_p
        

def compute_empirical_joint_posterior_distribution(X, Y, vec_PMF_sigma_X, PMF_sigma_Y):

    N_p = np.size(X)
    smear_Y = np.size(PMF_sigma_Y)
    P_XY = np.zeros((256, 256))

    for t in range(N_p):

        PMF_sigma_X = vec_PMF_sigma_X[t]
        smear_X = np.size(PMF_sigma_X)
        P_XY[X[t], Y[t]] += PMF_sigma_X[0] * PMF_sigma_Y[0]

        for k1 in range(1, smear_X):
            
            for k2 in range(1, smear_Y):

                P_XY[np.clip(X[t] + k1, 0, 255), np.clip(Y[t] + k2, 0, 255)] += PMF_sigma_X[k1] * PMF_sigma_Y[k2]
                P_XY[np.clip(X[t] + k1, 0, 255), np.clip(Y[t] - k2, 0, 255)] += PMF_sigma_X[k1] * PMF_sigma_Y[k2]
                P_XY[np.clip(X[t] - k1, 0, 255), np.clip(Y[t] + k2, 0, 255)] += PMF_sigma_X[k1] * PMF_sigma_Y[k2]
                P_XY[np.clip(X[t] - k1, 0, 255), np.clip(Y[t] - k2, 0, 255)] += PMF_sigma_X[k1] * PMF_sigma_Y[k2]

    return P_XY / np.sum(P_XY)



smear_width = 3

# Dimensions of the problem
N_w = 6 
N_d = 11
N_p = N_w * N_d
n_map_secs = 2
L = N_d * n_map_secs

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


# Generate an AR(1) process
# If the correlation coefficient alpha, then stationary distribution is Gaussian with variance sigma_AR^2 = sigma_w^2/(1-alpha^2)
# SSNR is calculated with respect to sigma_AR^2, and so sigma_w^2 is chosen to satisfy the requirements

mu = 128
sigma_AR = 5
sigma_AR_2_dB = 20*np.log10(sigma_AR)

SSNR_dB = 45
N_0_dB = sigma_AR_2_dB - SSNR_dB
N_0 = 10**(N_0_dB/10)

SINR_dB = 10
sigma_I_2_dB = sigma_AR_2_dB - SINR_dB
sigma_I = 10**(sigma_I_2_dB/20)

N_s = 20

# alpha = 0.99 # must satisty 0 <= alpha < 1

alphas = np.concatenate((np.arange(0.05, 0.95, 0.05), np.arange(0.95, 0.99, 0.01)))
Pe_NMI = np.zeros(np.size(alphas))
Pe_ENMI1D = np.zeros(np.size(alphas))
Pe_ENMI2D = np.zeros(np.size(alphas))

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

# Create a vec of PMFs for each of the X as it has different noise distribution
vec_sigma_S = np.sqrt(N_0 / vec_A)
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

  
for i in range(np.size(alphas)):
    Ne_NMI = 0
    Ne_ENMI1D = 0
    Ne_ENMI2D = 0
    alpha = alphas[i]
    for _ in tqdm(range(N_s)):
        
        sigma_w = sigma_AR * np.sqrt(1-alpha**2)
        signal_AR = np.zeros((N_w, L))
        
        for t in range(L):
            signal_w = np.random.normal(0, sigma_w, N_w)
            if t>0:
                signal_AR[:, t] = alpha * signal_AR[:, t-1] + signal_w
            else:
                signal_AR[:, 0] = np.random.normal(0, sigma_AR, N_w)
        
        map_0 = np.clip(mu + np.ravel(signal_AR[:, :N_d]) + sigma_I * np.random.randn(N_p), 0, 255).astype(np.uint8)
        map_1 = np.clip(mu + np.ravel(signal_AR[:, N_d:]) + sigma_I * np.random.randn(N_p), 0, 255).astype(np.uint8)
        
        captured_signal = np.clip(mu + np.ravel(signal_AR[:, :N_d]) + sigma_I * np.random.randn(N_p) 
                                    + vec_sigma_S * np.random.randn(N_p), 0, 255).astype(np.uint8)
        
        
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

    print(f"Alpha = {alphas[i]} | NMI = {Pe_NMI[i]} | ENMI1D = {Pe_ENMI1D[i]} | ENMI2D = {Pe_ENMI2D[i]}") 


plt.plot(alphas, Pe_NMI, label=r"$NMI$", ls=':', color='k')
plt.plot(alphas, Pe_ENMI1D, label=r"$ENMI_{1D}$", ls='--', color='k')
plt.plot(alphas, Pe_ENMI2D, label=r"$ENMI_{2D}$", color='k')
plt.grid()
plt.legend()
plt.xlabel(r"Correlation $(\alpha)$")
plt.ylabel(r"$P_e$")
plt.title(f"SSNR = {SSNR_dB}dB | SINR = {SINR_dB}dB")
plt.savefig('MI_AR1.png', dpi=400, bbox_inches='tight')

# Save the numpy arrays
np.save("alphas_MI.npy", alphas)
np.save("Pe_NMI_AR1.npy", Pe_NMI)
np.save("Pe_ENMI1D_AR1.npy", Pe_ENMI1D)
np.save("Pe_ENMI2D_AR1.npy", Pe_ENMI2D)