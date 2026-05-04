"""
Lab #3 - Channel equalization simulations.

Implements and compares:
  - Zero-forcing (ZF)
  - Linear MMSE
  - Non-linear sequential detection

The model is y_k = H^H x_k + g_k, with H of size Mt x Mr.
Run:
    python lab3_equalizers.py
It saves figures in the current folder.
"""

import numpy as np
import matplotlib.pyplot as plt


RNG = np.random.default_rng(7)
K = 10_000


H1 = np.array([
    [0.1, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 10.0],
], dtype=complex)

H2 = np.array([
    [1/2, 1 - 1j, 1j/2],
    [-1/2, 2j, -1j],
    [0, -1j/2, 2],
], dtype=complex)

H3 = np.array([
    [1j, -1, 3],
    [1/2, 1j/4, -1/2],
], dtype=complex)

H3_SWAPPED = H3[[1, 0], :]


def complex_awgn(mr: int, k: int, sigma_g2: float) -> np.ndarray:
    """Circular complex AWGN with E{|g|^2}=sigma_g2."""
    if sigma_g2 == 0:
        return np.zeros((mr, k), dtype=complex)
    return np.sqrt(sigma_g2 / 2) * (RNG.standard_normal((mr, k)) + 1j * RNG.standard_normal((mr, k)))


def generate_data(H: np.ndarray, alphabet: np.ndarray, sigma_g2: float, k: int = K):
    """Generate X and Y according to y = H^H x + g."""
    mt, mr = H.shape
    X = RNG.choice(alphabet, size=(mt, k)).astype(float)
    G = complex_awgn(mr, k, sigma_g2)
    Y = H.conj().T @ X + G
    return X, Y


def slicer(xhat: np.ndarray, alphabet: np.ndarray) -> np.ndarray:
    """Nearest-neighbour decision over a real alphabet."""
    xr = np.real(xhat)
    distances = np.abs(xr[..., None] - alphabet[None, None, :])
    indices = np.argmin(distances, axis=-1)
    return alphabet[indices]


def zf_equalizer(H: np.ndarray, Y: np.ndarray, alphabet: np.ndarray):
    """ZF equalizer: F_ZF = H^H (H H^H)^(-1), xhat = F_ZF^H y."""
    F = H.conj().T @ np.linalg.inv(H @ H.conj().T)
    Xhat = F.conj().T @ Y
    return slicer(Xhat, alphabet), F


def mmse_equalizer(H: np.ndarray, Y: np.ndarray, alphabet: np.ndarray, sigma_g2: float):
    """MMSE equalizer: F = (H^H H + sigma_g2/Es I)^(-1) H^H."""
    mr = H.shape[1]
    Es = float(np.mean(np.abs(alphabet) ** 2))
    F = np.linalg.inv(H.conj().T @ H + (sigma_g2 / Es) * np.eye(mr)) @ H.conj().T
    Xhat = F.conj().T @ Y
    return slicer(Xhat, alphabet), F


def sequential_equalizer(H: np.ndarray, Y: np.ndarray, alphabet: np.ndarray):
    """Sequential detector using H H^H = U^H U and z = U^{-H} H y."""
    mt = H.shape[0]
    # np.linalg.cholesky returns L with A = L L^H. We need U with A = U^H U, so U=L^H.
    U = np.linalg.cholesky(H @ H.conj().T).conj().T
    Z = np.linalg.solve(U.conj().T, H @ Y)

    Xtilde = np.zeros((mt, Y.shape[1]), dtype=float)
    # Detect each layer for all samples at once, from last to first.
    for i in range(mt - 1, -1, -1):
        residual = Z[i, :] - U[i, i + 1:] @ Xtilde[i + 1:, :]
        soft = residual / U[i, i]
        distances = np.abs(np.real(soft)[:, None] - alphabet[None, :])
        Xtilde[i, :] = alphabet[np.argmin(distances, axis=1)]
    return Xtilde, U


def error_probability(X: np.ndarray, Xtilde: np.ndarray) -> float:
    return float(np.mean(Xtilde != X))


def simulate_once(H: np.ndarray, alphabet: np.ndarray, sigma_g2: float):
    X, Y = generate_data(H, alphabet, sigma_g2)
    Xzf, Fzf = zf_equalizer(H, Y, alphabet)
    Xmmse, Fmmse = mmse_equalizer(H, Y, alphabet, sigma_g2)
    Xseq, U = sequential_equalizer(H, Y, alphabet)
    return {
        "ZF": error_probability(X, Xzf),
        "MMSE": error_probability(X, Xmmse),
        "Sequential": error_probability(X, Xseq),
        "FZF": Fzf,
        "FMMSE": Fmmse,
        "U": U,
    }


def plot_curves(title: str, sigmas: np.ndarray, results: dict, filename: str):
    plt.figure(figsize=(7, 4.5))
    floor = 1 / (K * 10)  # only for plotting zero-error points on a log axis
    for name, values in results.items():
        plt.semilogy(sigmas, np.maximum(values, floor), marker="o", label=name)
    plt.xlabel(r"Noise variance $\sigma_g^2$")
    plt.ylabel("Error probability")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def main():
    bpsk = np.array([-1, 1], dtype=float)
    pam4 = np.array([-3, -1, 1, 3], dtype=float)

    print("=== H1, sigma_g^2 = 1 ===")
    for name, alphabet in [("BPSK", bpsk), ("4-PAM", pam4)]:
        res = simulate_once(H1, alphabet, sigma_g2=1.0)
        print(f"\n{name}")
        print("Pe ZF         =", res["ZF"])
        print("Pe MMSE       =", res["MMSE"])
        print("Pe Sequential =", res["Sequential"])
        print("FZF =\n", np.round(res["FZF"], 6))
        print("FMMSE =\n", np.round(res["FMMSE"], 6))

    print("\n=== H2, BPSK, sigma_g^2 in [0,1] ===")
    sigmas_h2 = np.linspace(0, 1, 21)
    curves_h2 = {"ZF": [], "MMSE": [], "Sequential": []}
    for sg2 in sigmas_h2:
        res = simulate_once(H2, bpsk, sigma_g2=float(sg2))
        for key in curves_h2:
            curves_h2[key].append(res[key])
    for key in curves_h2:
        curves_h2[key] = np.array(curves_h2[key])
        print(key, curves_h2[key])
    plot_curves("H2, BPSK", sigmas_h2, curves_h2, "lab3_H2_BPSK_equalizers.png")

    print("\n=== H3 and row-swapped H3, BPSK, sigma_g^2 in (0,1] ===")
    sigmas_h3 = np.linspace(0.05, 1, 20)
    for H, label, filename in [
        (H3, "H3", "lab3_H3_BPSK_equalizers.png"),
        (H3_SWAPPED, "H3 row-swapped", "lab3_H3_swapped_BPSK_equalizers.png"),
    ]:
        curves = {"MMSE": [], "Sequential": []}
        for sg2 in sigmas_h3:
            X, Y = generate_data(H, bpsk, float(sg2))
            Xmmse, _ = mmse_equalizer(H, Y, bpsk, float(sg2))
            Xseq, _ = sequential_equalizer(H, Y, bpsk)
            curves["MMSE"].append(error_probability(X, Xmmse))
            curves["Sequential"].append(error_probability(X, Xseq))
        for key in curves:
            curves[key] = np.array(curves[key])
            print(label, key, curves[key])
        plot_curves(label + ", BPSK", sigmas_h3, curves, filename)

    print("\nFigures saved:")
    print("  lab3_H2_BPSK_equalizers.png")
    print("  lab3_H3_BPSK_equalizers.png")
    print("  lab3_H3_swapped_BPSK_equalizers.png")


if __name__ == "__main__":
    main()
