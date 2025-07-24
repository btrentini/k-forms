import torch


def generate_point_cloud_dataset(num_samples: int, num_points: int, noise_std: float = 0.01, device: str = 'cpu'):
    """
    Generates a synthetic dataset of 2D point clouds with two categories.
    Each point cloud has points along either a rising or falling diagonal,
    randomly shifted along the y-axis. Points falling outside [0,1]^2 are excluded.

    Args:
        num_samples (int): Number of point clouds.
        num_points (int): Number of points per point cloud (after filtering).
        noise_std (float): Std of Gaussian noise.
        device (str): Torch device.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (X, y)
            - X: (num_samples, num_points, 2) point clouds
            - y: (num_samples,) labels (1 for rising, 0 for falling diagonal)
    """
    X = torch.zeros((num_samples, num_points, 2), device=device)
    y = torch.zeros(num_samples, dtype=torch.long, device=device)

    for i in range(num_samples):
        label = torch.randint(0, 2, (1,), device=device).item()
        y[i] = label

        # Choose safe y-shift range
        max_shift = 0.5 - noise_std * 3
        y_shift = (2 * torch.rand(1, device=device) - 1) * max_shift  # Uniform in [-max_shift, max_shift]

        # Keep oversampling until we get enough in-bounds points
        points = []
        while len(points) < num_points:
            t = torch.rand(num_points * 2, 1, device=device)  # Oversample

            if label == 1:
                base = torch.hstack([t, t + y_shift])
            else:
                base = torch.hstack([t, 1 - t + y_shift])

            noise = torch.randn_like(base) * noise_std
            noisy = base + noise

            # Keep only in-bounds points
            mask = ((noisy >= 0) & (noisy <= 1)).all(dim=1)
            in_bounds = noisy[mask]

            # Accumulate until we have enough
            if in_bounds.shape[0] > 0:
                points.append(in_bounds)

            points = torch.cat(points, dim=0)
            if points.shape[0] > num_points:
                points = points[:num_points]

        X[i] = points

    return X, y



