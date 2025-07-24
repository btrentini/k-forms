import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.neighbors import NearestNeighbors
from math import comb


def compute_heat_kernel(X, k_NN):
    t = 1.0
    p, n = X.shape
    # Step 2: k-NN graph
    if p - 1 < k_NN:
        k_NN = p - 1
    nbrs = NearestNeighbors(n_neighbors=k_NN + 1).fit(X)  # +1 includes the point itself
    distances, indices = nbrs.kneighbors(X)

    # Step 3: Build heat kernel matrix
    heat_kernel = torch.zeros(p, p)

    for i in range(p):
        """
        How heat would diffuse from poin i to point j over time t.
        This favours local interactions, and is closely related to the Gaussian kernel.
        The kernel is symmetric and positive, giving rise to a weighted adjacency matrix for the graph.
        """
        for j_idx, j in enumerate(indices[i][1:]):  # Skip self-loop
            d2 = torch.norm(X[i] - X[j]) ** 2
            coef = 1 / ((4 * np.pi * t) ** (n / 2))
            heat_kernel[i, j] = coef * torch.exp(-d2 / (4 * t))
            heat_kernel[j, i] = heat_kernel[i, j]  # symmetry

    return heat_kernel


def compute_laplacian(X, k_NN):
    # Degree matrix
    # TODO: Sum over K-neighbours to make it sparse instead of axis = 1]
    heat_kernel = compute_heat_kernel(X, k_NN)
    deg = torch.diag(heat_kernel.sum(dim=1))

    # Unnormalised Laplacian
    # TODO: Over colums that are neighbour
    p = X.shape[0]
    L = (1 / p) * (deg - heat_kernel)
    return L


def dpdp_lookup_matrix(L, X):
    """
    Optimized lookup matrix calculation using vectorization.

    Args:
      L: Laplacian matrix of shape (p, p).
      X: Point cloud data of shape (p, n).

    Returns:
      A tensor of shape (p, n, n) resulting from the computation.
    """
    # 1. Pre-calculate L @ X, shape remains (p, n)
    dp = L @ X

    # Calculate the first two terms: P*dp' + dp*P'
    # 'pi, qj -> pqij' would create a (p,p,n,n) tensor, which is too large.
    # We use broadcasting and einsum to compute the terms directly.
    # 'pi,pj->pij' computes the element-wise product of every column i of P
    # with every column j of dp, resulting in a (p, n, n) tensor.
    pdp = torch.einsum('pi,pj->pij', X, dp)

    # The second term is simply the transpose of the first w.r.t the last two dims.
    pdp_T = pdp.permute(0, 2, 1)

    # 3. Calculate the third term: L @ (P[:,i] * P[:,j])
    # First, get the outer product of P's columns: (P[:,i] * P[:,j])
    had_p2 = torch.einsum('pi,pj->pij', X, X)

    # Then, apply L to each p-dimensional vector in had_p2.
    # 'ap,pij->aij' multiplies matrix L (ap) with tensor had_p2 (pij)
    # and sums over the common dimension 'p'.
    L_had_p2 = torch.einsum('ap,pij->aij', L, had_p2)

    # 4. Combine the results
    dpdp = pdp + pdp_T - L_had_p2

    return dpdp


def dpdp_determinant(dpdp, I, J):
    """
    dpdp: lookup of the pullbacked basis at each point (p x n x n)
    """
    k = len(I)
    p = dpdp.shape[0]
    pG = torch.zeros(p, k, k)
    for i in range(k):
        for j in range(k):
            pG[:, i, j] = dpdp[:, I[i], J[j]]

    det_PG = torch.zeros(p)
    for i in range(p):
        det_PG[i] = torch.linalg.det(pG[i])

    return det_PG


def generate_multiindices(n: int, k: int) -> torch.Tensor:
    """
    Generate all lexicographically ordered multiindices choosing k out of n indices.

    Args:
        n: Total number of indices to choose from (0 to n-1)
        k: Number of indices to choose

    Returns:
        torch.Tensor: A tensor of shape (num_combinations, k) containing all combinations
                     in lexicographic order
    """
    if k > n or k < 0 or n < 0:
        raise ValueError("Invalid parameters: k must be between 0 and n")

    if k == 0:
        return torch.empty((1, 0), dtype=torch.long)

    # Calculate total number of combinations: C(n, k) = n! / (k! * (n-k)!)
    total_combinations = comb(n, k)

    # Initialize result tensor
    result = torch.zeros((total_combinations, k), dtype=torch.long)

    # Generate combinations using iterative approach
    current_combination = list(range(k))  # Start with [0, 1, 2, ..., k-1]

    for i in range(total_combinations):
        # Store current combination
        result[i] = torch.tensor(current_combination)

        # Generate next combination in lexicographic order
        if i < total_combinations - 1:  # Not the last combination
            # Find the rightmost index that can be incremented
            pos = k - 1
            while pos >= 0 and current_combination[pos] == n - k + pos:
                pos -= 1

            # Increment the found position
            current_combination[pos] += 1

            # Reset all positions to the right
            for j in range(pos + 1, k):
                current_combination[j] = current_combination[j - 1] + 1

    return result


def compute_detmultiindices(dpdp, multiindices_k, p, n, k):
    """
    p: point cloud size
    n: dimension of space of the points
    k: specifying the k-form
    """

    combinations = comb(n, k)

    full_PG = torch.zeros(p, combinations, combinations)
    for i in range(combinations):
        I = multiindices_k[i]
        for j in range(combinations):
            J = multiindices_k[j]
            full_PG[:, i, j] = dpdp_determinant(dpdp, I, J)

    return full_PG


class GeometricPointCloud(Dataset):
    """
    A PyTorch Dataset class for handling point cloud data.

    For each point cloud in the input dataset, this class returns a pair:
    the original point cloud and a version of it that has been modified by a
    provided transformation function. This is particularly useful for
    self-supervised learning models or for data augmentation pipelines where
    the original data is also needed.

    Args:
        point_clouds (list or np.ndarray): A list or array of point clouds.
            Each point cloud should be a NumPy array of shape (num_points, num_features).
        transform_fn (callable): A function that takes a single point cloud
            (NumPy array) and returns its transformed version.
    """

    def __init__(self, X, X_labels, transform_fn):
        if not callable(transform_fn):
            raise TypeError("The provided transform_fn must be a callable function.")

        self.point_clouds = X
        self.labels = X_labels
        self.transform_fn = transform_fn

    def __len__(self):
        """Returns the total number of point clouds in the dataset."""
        return len(self.point_clouds)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing two PyTorch tensors:
                   - The original point cloud.
                   - The transformed point cloud.
        """
        # Retrieve the original point cloud
        original_pc = self.point_clouds[idx]

        # Apply the transformation to create the second version
        # We make a copy to ensure the original data is not modified in place
        transformed_pc = self.transform_fn(torch.clone(original_pc))

        return (transformed_pc, original_pc), self.labels[idx]


def geometrify_point_cloud(X, k, k_NN):
    if X.shape[0] == 0:
        return None
    L = compute_laplacian(X, k_NN)
    dpdp = dpdp_lookup_matrix(L, X)
    n = X.shape[1]
    p = X.shape[0]
    multiindices_k = generate_multiindices(n, k)
    PG = compute_detmultiindices(dpdp, multiindices_k, p, n, k)
    return PG


# geometrifier = partial(geometrify_point_cloud, k=k)
# P, P_label = GeometricPointCloud(X, X_labels, geometrifier)[7]
