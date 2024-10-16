import numpy as np
import matplotlib.pyplot as plt

def initialize_states(steps, alpha, beta):
    a = np.zeros((2 * steps + 5, steps + 1), dtype=complex)
    b = np.zeros((2 * steps + 5, steps + 1), dtype=complex)
    a[steps, 0] = np.cos(0.5 * alpha)
    b[steps, 0] = np.exp(1j * beta) * np.sin(0.5 * alpha)
    return a, b

def update_amplitudes(a, b, q, theta, phi, steps):
    DP = np.zeros(2 * steps + 5)
    sigma = np.zeros(steps + 1)
    SE = np.zeros(steps + 1)
    
    for t in range(1, steps + 1):
        A = 0
        gamma = 0
        jqm = 0
        jm = 0
        
        for j in range(-t, t + 1):
            j_idx = j + steps
            a[j_idx, t] = np.sqrt(q) * a[j_idx - 1, t - 1] + np.sqrt(1 - q) * np.exp(1j * theta) * b[j_idx - 1, t - 1]
            b[j_idx, t] = np.sqrt(1 - q) * np.exp(1j * phi) * a[j_idx + 1, t - 1] - np.sqrt(q) * np.exp(1j * (theta + phi)) * b[j_idx + 1, t - 1]
            
            A += np.abs(a[j_idx, t]) ** 2
            gamma += a[j_idx, t] * np.conjugate(b[j_idx, t])
            
            if t == steps:
                DP[j_idx] = np.abs(a[j_idx, t]) ** 2 + np.abs(b[j_idx, t]) ** 2
            
            jqm += j ** 2 * (np.abs(a[j_idx, t]) ** 2 + np.abs(b[j_idx, t]) ** 2)
            jm += j * (np.abs(a[j_idx, t]) ** 2 + np.abs(b[j_idx, t]) ** 2)
        
        sigma[t] = np.sqrt(jqm - jm ** 2)
        lambda_ = np.sqrt(0.25 - A * (1 - A) + gamma * np.conjugate(gamma))
        lambda_real = np.real(lambda_)
        SE[t] = -(0.5 + lambda_real) * np.log2(0.5 + lambda_real) - (0.5 - lambda_real) * np.log2(0.5 - lambda_real)

    return DP, sigma, SE

def plot_results(DP, sigma, SE, steps):
    Pxj = [(j - steps, DP[j]) for j in range(2 * steps + 1)]
    SExt = [(t, SE[t]) for t in range(1, steps + 1)]
    sigmaxt = [(t, sigma[t]) for t in range(1, steps + 1)]

    plt.figure()
    plt.plot(*zip(*Pxj))
    plt.fill_between(*zip(*Pxj), color='blue', alpha=0.3)
    plt.xlabel("j")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(*zip(*sigmaxt))
    plt.xlabel("t")
    plt.ylabel("Dispersion")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(*zip(*SExt))
    plt.xlabel("t")
    plt.ylabel("SE")
    plt.ylim([0, 1])
    plt.grid(True)
    plt.show()

# Main code
steps = 100
alpha = 0
beta = 0
q = 0.5
theta = 0
phi = 0

a, b = initialize_states(steps, alpha, beta)
DP, sigma, SE = update_amplitudes(a, b, q, theta, phi, steps)
plot_results(DP, sigma, SE, steps)
