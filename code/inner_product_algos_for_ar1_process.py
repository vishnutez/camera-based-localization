import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

N_w = 6
N_d = 11
n_map_secs = 2
L = N_d * n_map_secs
N_p = N_w * N_d

# Parameters of the camera model
f = 0.0367
h = 60
theta = 36 
theta_rad = theta * np.pi / 180
s = 20

# Generate an AR(1) process

# If the AR-coefficient is alpha, then the stationary distribution has power with sigma_AR^2 = sigma_w^2 / (1-alpha^2)

# SSNR is calculated with respect to sigma_AR^2, and so sigma_w^2 is chosen to satisfy the requirements

mu = 128
sigma_AR = 5
sigma_AR_2_dB = 20*np.log10(sigma_AR)

SSNR_dB = 45
sigma_S_2_dB = sigma_AR_2_dB - SSNR_dB
N0 = 10**(sigma_S_2_dB / 10)  # Noise spectral density

A = np.zeros([N_w, N_d])
for j in range(N_d):
    A[:, j] = f**2 * h * s / (2 * np.cos(theta_rad)) * ( 1/((j+1) * s * np.cos(theta_rad) + h * np.sin(theta_rad))**2 - 1/((j+2) * s * np.cos(theta_rad) + h * np.sin(theta_rad))**2 )
vec_A = np.ravel(A)

SINR_dB = 3
sigma_I_2_dB = sigma_AR_2_dB - SINR_dB
sigma_I = 10**(sigma_I_2_dB / 20)

# Set alpha = 0.99 must satisty 0 <= alpha < 1 for AR(1) process
alphas = np.concatenate((np.arange(0.05, 0.95, 0.05), np.arange(0.95, 0.99, 0.01)))
Pe_GIP2D = np.zeros(np.size(alphas))
Pe_GIP1D = np.zeros(np.size(alphas))
Pe_SIP = np.zeros(np.size(alphas))

N_s = 1000

sigma_S = np.sqrt(N0/vec_A)  # Vector of sensor noise levels

for i in range(np.size(alphas)):

    # Gather and initialize
    alpha = alphas[i]
    Ne_GIP2D = 0
    Ne_GIP1D = 0
    Ne_SIP = 0
    
    for _ in tqdm(range(N_s)):

        # Generate AR-1 process over the depth of the road (N_d)
        sigma_w = sigma_AR * np.sqrt(1-alpha**2)
        signal_AR = np.zeros((N_w, L))
        for t in range(L):
            signal_w = np.random.normal(0, sigma_w, N_w)
            if t > 0:
                signal_AR[:, t] = alpha * signal_AR[:, t-1] + signal_w
            else:
                signal_AR[:, 0] = np.random.normal(0, sigma_AR, N_w)

        # Generate map sections with ground truth noise
        map_0 = np.clip(mu + np.ravel(signal_AR[:, :N_d]) + sigma_I * np.random.randn(N_p), 0, 255).astype(np.uint8)
        map_1 = np.clip(mu + np.ravel(signal_AR[:, N_d:]) + sigma_I * np.random.randn(N_p), 0, 255).astype(np.uint8)
        
        captured_signal = np.clip(mu + np.ravel(signal_AR[:, :N_d]) + sigma_I * np.random.randn(N_p)
                                    + sigma_S * np.random.randn(N_p), 0, 255).astype(np.uint8)     
        
        # Compute the distances in the induced inner product spaces
        GIP2D_0 = np.linalg.norm((captured_signal-map_0)**2/(sigma_S**2 + 2* sigma_I**2))
        GIP2D_1 = np.linalg.norm((captured_signal-map_1)**2/(sigma_S**2 + 2* sigma_I**2))
        GIP1D_0 = np.linalg.norm((captured_signal-map_0)**2/sigma_S**2)
        GIP1D_1 = np.linalg.norm((captured_signal-map_1)**2/sigma_S**2)
        SIP_0 = np.linalg.norm((captured_signal-map_0)**2)
        SIP_1 = np.linalg.norm((captured_signal-map_1)**2)

        # Count the number of errors
        Ne_GIP2D += (GIP2D_0 > GIP2D_1)
        Ne_GIP1D += (GIP1D_0 > GIP1D_1)
        Ne_SIP += (SIP_0 > SIP_1)
    
    Pe_GIP2D[i] = Ne_GIP2D / N_s
    Pe_GIP1D[i] = Ne_GIP1D / N_s
    Pe_SIP[i] = Ne_SIP / N_s

    print(f"Alpha = {alphas[i]} | SIP = {Pe_SIP[i]} | GIP1D = {Pe_GIP1D[i]} | GIP2D = {Pe_GIP2D[i]}") 
    
plt.plot(alphas, Pe_SIP, label=r"$SIP$", linestyle=":", color="k")
plt.plot(alphas, Pe_GIP1D, label=r"$GIP_{1D}$", linestyle="--", color="k")
plt.plot(alphas, Pe_GIP2D, label=r"$GIP_{2D}$", color="k")
plt.xlabel(r"AR-1 coefficient $(\alpha)$")
plt.ylabel("Probability of misclassification")
plt.title(f"SSNR = {SSNR_dB}dB | SINR = {SINR_dB}dB")
plt.legend()
plt.grid()
plt.savefig('IP_AR1.png', dpi=400, bbox_inches='tight')

np.save("alphas_IP.npy", alphas)
np.save("Pe_SIP_AR1.npy", Pe_SIP)
np.save("Pe_GIP1D_AR1.npy", Pe_GIP1D)
np.save("Pe_GIP2D_AR1.npy", Pe_GIP2D)
