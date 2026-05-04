"""
Lab #3 - Interference cancellation and hidden-message decoding.

Input file: data_rx.txt
Model:
    r[k] = B^T (s[k] + x[k])
    x[k+1] = A x[k] + u[k],  u[k] ~ N(0, sigma_u^2 I)
    G(D) = [D^2 + 1, D^2 + D + 1]

This script uses a trellis/Viterbi detector whose branch metric is the innovation
of the reconstructed interference sequence. It avoids treating the unknown coded
symbols as Gaussian noise.

Run:
    python lab3_interference_decode.py data_rx.txt
"""

from __future__ import annotations

import sys
import numpy as np


A = np.array([[0.90, 0.12],
              [0.08, 0.85]], dtype=float)

BT = np.array([[-0.12,  0.15],
               [ 0.40, -0.15]], dtype=float)

SIGMA_U = 0.15


def load_received(filename: str) -> np.ndarray:
    """Load concatenated r[k] values and reshape into a 2 x N matrix."""
    values = np.loadtxt(filename)
    if values.size % 2 != 0:
        raise ValueError("The received vector must contain an even number of samples.")
    return values.reshape((2, -1), order="F")


def encoded_symbols(input_bit: int, state: tuple[int, int]) -> np.ndarray:
    """
    Convolutional encoder G(D) = [D^2+1, D^2+D+1].

    state = (b[k-1], b[k-2]).
    Output bits:
        c1[k] = b[k] xor b[k-2]
        c2[k] = b[k] xor b[k-1] xor b[k-2]
    BPSK mapping used by the data: 0 -> -1, 1 -> +1.
    """
    b1, b2 = state
    c1 = input_bit ^ b2
    c2 = input_bit ^ b1 ^ b2
    return np.array([2 * c1 - 1, 2 * c2 - 1], dtype=float)


def viterbi_with_interference(v: np.ndarray):
    """
    Decode information bits from v[k] = s[k] + x[k].

    For each candidate branch, s[k] is known, hence x[k] = v[k] - s[k].
    The branch metric is ||x[k] - A x[k-1]||^2 / sigma_u^2, with x[0]=0.
    """
    n = v.shape[1]
    states = [(0, 0), (0, 1), (1, 0), (1, 1)]
    inf = 1e100

    metric = {state: inf for state in states}
    prev_x = {state: None for state in states}
    metric[(0, 0)] = 0.0  # zero encoder memory before the first bit

    backpointers: list[dict[tuple[int, int], tuple[tuple[int, int], int]]] = []

    for k in range(n):
        new_metric = {state: inf for state in states}
        new_prev_x = {state: None for state in states}
        bp = {}

        for state in states:
            if metric[state] >= inf / 2:
                continue

            for bit in (0, 1):
                s_k = encoded_symbols(bit, state)
                x_k = v[:, k] - s_k

                if k == 0:
                    # x[0] is known to be exactly zero. A large weight enforces this.
                    branch_metric = 1e6 * float(x_k @ x_k)
                else:
                    innovation = x_k - A @ prev_x[state]
                    branch_metric = float(innovation @ innovation) / (SIGMA_U ** 2)

                next_state = (bit, state[0])
                candidate_metric = metric[state] + branch_metric
                if candidate_metric < new_metric[next_state]:
                    new_metric[next_state] = candidate_metric
                    new_prev_x[next_state] = x_k
                    bp[next_state] = (state, bit)

        metric = new_metric
        prev_x = new_prev_x
        backpointers.append(bp)

    end_state = min(states, key=lambda st: metric[st])
    bits = np.zeros(n, dtype=np.uint8)
    state = end_state

    for k in range(n - 1, -1, -1):
        previous_state, bit = backpointers[k][state]
        bits[k] = bit
        state = previous_state

    return bits, metric[end_state], end_state


def bits_to_ascii(bits: np.ndarray) -> str:
    """Convert MSB-first bits into ASCII. Extra tail bits are ignored."""
    usable = (len(bits) // 8) * 8
    chars = []
    for i in range(0, usable, 8):
        value = 0
        for bit in bits[i:i + 8]:
            value = (value << 1) | int(bit)
        chars.append(chr(value))
    return "".join(chars)


def main(filename: str = "data_rx.txt"):
    R = load_received(filename)
    v = np.linalg.inv(BT) @ R  # v[k] = s[k] + x[k]

    bits, metric, end_state = viterbi_with_interference(v)
    message = bits_to_ascii(bits)

    print(f"Loaded {R.shape[1]} received vectors.")
    print(f"Decoded {len(bits)} information bits; {len(bits) % 8} tail bit(s) ignored for ASCII.")
    print(f"Final trellis state: {end_state}")
    print(f"Path metric: {metric:.3f}")
    print("\nRecovered message:\n")
    print(message)

    np.savetxt("decoded_bits.txt", bits, fmt="%d")
    with open("decoded_message.txt", "w", encoding="utf-8") as f:
        f.write(message + "\n")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data_rx.txt"
    main(input_file)
