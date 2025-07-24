import torch
import numpy as np
import matplotlib.pyplot as plt
import noisy_lines_dataset
from kgram_models import Kgrams
from torch.utils.data import DataLoader
from functools import partial
from geometric_point_cloud import GeometricPointCloud, geometrify_point_cloud


def visualize_dataset(X: torch.Tensor, y: torch.Tensor, num_to_show: int = 8):
    """
    Visualizes a selection of point clouds from the generated dataset.

    Args:
        X (torch.Tensor): The point cloud data (from generate_point_cloud_dataset).
        y (torch.Tensor): The labels for the point clouds.
        num_to_show (int, optional): The number of examples to plot. Defaults to 8.
    """
    if num_to_show > len(X):
        print(f"Warning: num_to_show ({num_to_show}) is greater than the number of samples ({len(X)}). Showing all samples.")
        num_to_show = len(X)

    # Create a grid of subplots
    # We try to make the grid as square as possible
    cols = int(np.ceil(np.sqrt(num_to_show)))
    rows = int(np.ceil(num_to_show / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()  # Flatten to make indexing easier

    # Select random indices to display
    indices = torch.randperm(len(X))[:num_to_show]

    for i, ax_idx in enumerate(range(num_to_show)):
        ax = axes[ax_idx]
        sample_idx = indices[i]

        # Move tensor to CPU and convert to numpy for plotting
        points = X[sample_idx].cpu().numpy()
        label = y[sample_idx].cpu().item()

        # Scatter plot for the points
        ax.scatter(points[:, 0], points[:, 1], alpha=0.7)

        # Set titles and limits
        title = "Category: Diagonal Up (y=1)" if label == 1 else "Category: Diagonal Down (y=0)"
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.5)

    # Hide any unused subplots
    for i in range(num_to_show, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def unit_square_coords(n: int = 101) -> np.ndarray:
    """
    Return an (n*n, 2) array of (x, y) points evenly spaced on a unit square.

    Parameters
    ----------
    n : int, optional
        Number of points along each axis (default 101 gives 1 % spacing).

    Returns
    -------
    coords : ndarray, shape (n*n, 2)
        Flattened array where coords[i] = [x_i, y_i].
    """
    x = np.linspace(0.0, 1.0, n)        # 1-D array of x-coordinates
    y = np.linspace(0.0, 1.0, n)        # 1-D array of y-coordinates
    xv, yv = np.meshgrid(x, y, indexing="xy")  # 2-D grids
    coords = np.column_stack((xv.ravel(), yv.ravel()))
    return coords.astype(np.float32)


def plot_vector_fields(model, k, k_form_selection=0):
    vector_scale = 1.0

    # Example: each row is an (x, y) coordinate
    coords = unit_square_coords(10)

    k_form = model.evaluate_kforms(torch.from_numpy(coords))[:, :, k_form_selection]
    k_form = np.squeeze(k_form.detach().numpy())
    print(k_form.shape)
    # Vector components
    U = coords[:, 0]  # x-components
    V = coords[:, 1]  # y-components

    # Create the plot
    plt.figure(figsize=(6, 6))
    # plt.quiver(origin_x, origin_y, U, V, angles='xy', scale_units='xy', scale=1/vector_scale)
    # plt.quiver(coords[:, 0], coords[:, 1], k_form[:,0], k_form[:, 1], angles='xy', scale_units='xy', scale=1/vector_scale)
    plt.quiver(coords[:, 0], coords[:, 1], k_form[:, 0], k_form[:, 1], angles='xy', scale_units='xy')

    # Set grid, labels, and equal aspect ratio
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'{k_form_selection}th {k}-form')
    plt.xlim(U.min() - 1, U.max() + 1)
    plt.ylim(V.min() - 1, V.max() + 1)
    plt.axhline(0)  # horizontal axis line
    plt.axvline(0)  # vertical axis line
    plt.gca().set_aspect('equal')

    plt.show()


def train(dataset, model):
    BATCH_SIZE = 7
    N_EPOCHS = 15
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # drop_last    = True,
        # pin_memory   = (DEVICE == 'cuda'),
        # num_workers  = 4               # tweak for your machine
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for feats, labels in train_loader:
            # feats, labels = feats.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(feats[0], feats[1])
            # print(feats[0].shape, feats[1].shape)
            # print(outputs.shape, labels.shape)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        print(f"[{epoch:02}/{N_EPOCHS}] training loss: {running_loss / total:.4f}")

    print("Training finished.")

    return model, running_loss / total


# --- Main execution block to demonstrate the functions ---
if __name__ == "__main__":
    # --- Parameters ---
    N_SAMPLES = 10  # Total number of point clouds in the dataset
    N_POINTS = 50     # Number of points in each cloud
    NOISE = 0.025      # Standard deviation of the noise

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available

    # --- Generate Dataset ---
    print(f"Generating dataset on device: {DEVICE}...")
    X, X_labels = noisy_lines_dataset.generate_point_cloud_dataset(
        num_samples=N_SAMPLES,
        num_points=N_POINTS,
        noise_std=NOISE,
        device=DEVICE
    )
    print("Dataset generation complete.")
    print(f"Shape of point cloud data (X): {X.shape}")
    print(f"Shape of labels (y): {X_labels.shape}")
    print("-" * 30)

    # --- Visualize a few examples ---
    print("Visualizing 8 random examples from the dataset...")
    visualize_dataset(X, X_labels, num_to_show=8)

    hidden_dim = 128
    n_classes = 1  # Irrelevant right now, we are not using the classification head atm
    l = 2
    n = X.shape[-1]
    k = 1

    model = Kgrams(l, n, k, hidden_dim, n_classes)

    geometrifier = partial(geometrify_point_cloud, k=k, k_NN=6)
    dataset = GeometricPointCloud(X, X_labels, geometrifier)
    model, loss = train(dataset, model)

    plot_vector_fields(model, k, k_form_selection=0)
