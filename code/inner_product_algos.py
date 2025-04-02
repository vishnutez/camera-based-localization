import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Dimensions of the problem
N_w = 6 
N_d = 11
N_p = N_w * N_d

# Initialize camera/mount parameters
f = 0.0367
h = 60
theta = 36 
theta_rad = theta * np.pi/180
s = 20

# Initialize intensity variation signal parameters
sigma = 5
sigma_2 = sigma**2
mu = 128

# Set intrinstic noise power levels
SINR_dB = 3
SINR = 10**(SINR_dB/10)
sigma_I_2 = sigma_2/SINR
sigma_I = np.sqrt(sigma_I_2)

# Compute the transformed tile areas for the above mounted camera
A = np.zeros([N_w, N_d])

for j in range(N_d):

    A[:, j] = f**2 * h * s / (2 * np.cos(theta_rad)) \
                * ( 1/((j+1) * s * np.cos(theta_rad) + h * np.sin(theta_rad))**2 \
                   - 1/((j+2) * s * np.cos(theta_rad) + h * np.sin(theta_rad))**2 )

vec_A = np.ravel(A)


# Set sensor noise levels to plot the results
SSNR_dB = np.arange(60, 10, -2)
SSNR = 10**(SSNR_dB/10)
N_0 = sigma**2/SSNR
n_SNR_values = np.size(SSNR_dB)


Pe_GIP2D = np.zeros(n_SNR_values)
Pe_GIP1D = np.zeros(n_SNR_values)
Pe_SIP = np.zeros(n_SNR_values)
N_s = 1000  # Number of statistical samples

for i in range(n_SNR_values):
    Ne_GIP2D = 0
    Ne_GIP1D = 0
    Ne_SIP = 0
    for _ in tqdm(range(N_s)):

        # Get the vector of sensor noise sigma_S for each tile using the transformation induced by the focal plane geometry
        sigma_S = np.sqrt(N_0[i]/vec_A)

        # Generate noise for observed signal
        z_S = sigma_S * np.random.randn(N_p)
        z_I = sigma_I * np.random.randn(N_p)

        # Generate intrinstic noise for the GT signals
        z_I_0 = sigma_I * np.random.randn(N_p)
        z_I_1 = sigma_I * np.random.normal(0, 1, N_p)

        # Generate underlying signals corresponding to the two distinct map sections
        signal_0 = mu + sigma * np.random.randn(N_p)
        signal_1 = mu + sigma * np.random.randn(N_p)

        # Generate map sections
        map_0 = np.clip(signal_0 + z_I_0, 0, 255).astype(np.uint8)
        map_1 = np.clip(signal_1 + z_I_1, 0, 255).astype(np.uint8)

        # Generate an instance of captured signal
        captured_signal = np.clip(signal_0 + z_I + z_S, 0, 255).astype(np.uint8)  # GT true signal: 0

        # Compute distances using the designed inner products
        GIP2D_0 = np.linalg.norm((captured_signal-map_0)**2/(sigma_S**2 + 2* sigma_I**2))
        GIP2D_1 = np.linalg.norm((captured_signal-map_1)**2/(sigma_S**2 + 2* sigma_I**2))

        GIP1D_0 = np.linalg.norm((captured_signal-map_0)**2/sigma_S**2)
        GIP1D_1 = np.linalg.norm((captured_signal-map_1)**2/sigma_S**2)

        SIP_0 = np.linalg.norm((captured_signal-map_0)**2)
        SIP_1 = np.linalg.norm((captured_signal-map_1)**2)

        # Count the number of misclassifications
        Ne_GIP2D += (GIP2D_0 > GIP2D_1)
        Ne_GIP1D += (GIP1D_0 > GIP1D_1)
        Ne_SIP += (SIP_0 > SIP_1)

    # Get the probability of error: Pe for each of the classifier
    Pe_GIP2D[i] = Ne_GIP2D / N_s
    Pe_GIP1D[i] = Ne_GIP1D / N_s
    Pe_SIP[i] = Ne_SIP / N_s

    print(f"SSNR = {SSNR_dB[i]}dB | SIP = {Pe_SIP[i]} | GIP1D = {Pe_GIP1D[i]} | GIP2D = {Pe_GIP2D[i]}")
    
# Plot the results vs N_0 (SSNR)
plt.semilogx(N_0, Pe_SIP, "k", linestyle=":", label=r"$SIP$")    
plt.semilogx(N_0, Pe_GIP1D, "k", linestyle="--", label=r"$GIP_{1D}$")
plt.semilogx(N_0, Pe_GIP2D, "k", label=r"$GIP_{2D}$")
plt.legend()
plt.grid()
plt.xlabel(r"$N_0$")
plt.ylabel("Probability of misclassification")
plt.title(f"SINR = {SINR_dB}dB")
plt.savefig('IP.png', dpi=400, bbox_inches='tight')

# Save the files
np.save("N_0_IP.npy", N_0)
np.save("Pe_SIP.npy", Pe_SIP)
np.save("Pe_GIP1D.npy", Pe_GIP1D)
np.save("Pe_GIP2D.npy", Pe_GIP2D)