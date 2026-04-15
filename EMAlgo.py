import argparse

import numpy as np

from Utils import parse_dataset

EPSILON = 1e-10


def initialize_parameters(
    num_class: int,
    n_r: int,
    n_c: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create initial Bernoulli parameters and mixture weights."""
    pixel_prob = np.clip(np.random.rand(num_class, n_r, n_c), EPSILON,
                         1 - EPSILON)
    mixing_weight = np.random.rand(num_class)
    mixing_weight /= mixing_weight.sum()
    return pixel_prob, mixing_weight


def expectation_step(
    data: np.ndarray,
    pixel_prob: np.ndarray,
    mixing_weight: np.ndarray,
) -> np.ndarray:
    """Compute posterior responsibilities in log-space for stability."""
    log_pixel_prob = np.log(np.clip(pixel_prob, EPSILON, 1 - EPSILON))
    log_inv_pixel_prob = np.log(np.clip(1 - pixel_prob, EPSILON, 1 - EPSILON))
    flattened_data = data.reshape(data.shape[0], -1)
    flattened_log_prob = log_pixel_prob.reshape(pixel_prob.shape[0], -1)
    flattened_log_inv_prob = log_inv_pixel_prob.reshape(
        pixel_prob.shape[0], -1)

    log_prob = (flattened_data @ flattened_log_prob.T +
                (1 - flattened_data) @ flattened_log_inv_prob.T +
                np.log(np.clip(mixing_weight, EPSILON, None)))
    max_log_prob = np.max(log_prob, axis=1, keepdims=True)
    stabilized_prob = np.exp(log_prob - max_log_prob)
    gamma = stabilized_prob / stabilized_prob.sum(axis=1, keepdims=True)
    return gamma


def maximization_step(data: np.ndarray,
                      gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Update mixture weights and Bernoulli parameters."""
    sum_gamma = gamma.sum(axis=0)
    mixing_weight = sum_gamma / data.shape[0]
    weighted_sum = np.tensordot(gamma.T, data, axes=(1, 0))
    pixel_prob = weighted_sum / sum_gamma[:, np.newaxis, np.newaxis]
    pixel_prob = np.clip(pixel_prob, EPSILON, 1 - EPSILON)
    return pixel_prob, mixing_weight


def run_em(
    data: np.ndarray,
    num_class: int,
    epoch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the EM iterations for a Bernoulli mixture model."""
    _, n_r, n_c = data.shape
    pixel_prob, mixing_weight = initialize_parameters(num_class, n_r, n_c)
    gamma = np.zeros((data.shape[0], num_class))

    for _ in range(epoch):
        gamma = expectation_step(data, pixel_prob, mixing_weight)
        pixel_prob, mixing_weight = maximization_step(data, gamma)

    return pixel_prob, mixing_weight, gamma


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("img_filepath", type=str)
    parser.add_argument("label_filepath", type=str)
    args = parser.parse_args()

    num_class = 10
    threshold = 128
    epoch = 1000
    _, _, _, imgs, _, labels = parse_dataset(args.img_filepath,
                                             args.label_filepath)
    data = (imgs > threshold).astype(np.float64)
    pixel_prob, mixing_weight, gamma = run_em(data, num_class, epoch)


if __name__ == "__main__":
    main()
