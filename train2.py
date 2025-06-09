# Updated code with trajectory and other analyses, fixing drift_magnitude error
import torch
import torchvision
from pathlib import Path
import pickle, gzip, math, os, time, shutil, matplotlib as mpl, numpy as np, matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance

# Create visualization directory
trex_viz_path = Path("bullseye")
trex_viz_path.mkdir(exist_ok=True)

# Load and preprocess data
try:
    df = pd.read_csv('DatasaurusDozen.tsv', sep='\t')
except FileNotFoundError:
    raise FileNotFoundError("Please ensure 'DatasaurusDozen.tsv' is in the working directory.")
datasaurus = np.asarray(df[df['dataset'] == 'bullseye'][['x', 'y']].values, dtype=float).reshape(-1, 2) # can be dino, star, bullseye, circle, | x_shape, star, high_lines, dots, circle, bullseye, slant_up, slant_down, wide_lines
plt.figure(figsize=[6, 6])
plt.scatter(datasaurus[:, 0], datasaurus[:, 1])
plt.axis('off')
plt.savefig(trex_viz_path / "bullseye{}.pdf".format("original"))
plt.close()

# Visualize noising schedule
alpha_min = 0.95
alpha_max = 0.9999
T = 50
alpha = torch.linspace(alpha_max, alpha_min, T)
alpha_bar = torch.cumprod(alpha, dim=-1).reshape(-1, 1)
plt.figure(figsize=[6, 6])
plt.plot(torch.arange(T), alpha_bar)
plt.xlabel('Timestep')
plt.ylim(0, 1.05)
plt.savefig(trex_viz_path / "alpha_bar_original.pdf")
plt.close()

def forward_noise_dataset(dataset, n_steps=1000, beta_min=0.0001, beta_max=0.02):
    beta = torch.linspace(beta_min, beta_max, n_steps)
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    s = dataset
    series = [s]
    for t in range(n_steps):
        noise = torch.randn_like(dataset)
        series.append(series[-1] * alpha_bar[t].sqrt() + (1. - alpha[t]).sqrt() * noise)
    return series

# Prepare data
datasaurus_tensor = torch.tensor(datasaurus, dtype=torch.float)
datasaurus_tensor = (datasaurus_tensor - datasaurus_tensor.mean()) / datasaurus_tensor.std()
trex_viz_input = torch.cat([datasaurus_tensor] * 6, dim=0)
datasaurus_series = forward_noise_dataset(trex_viz_input, beta_min=(1 - 0.9999), beta_max=(1 - 0.95), n_steps=50)

def scatters(cols, rows, datasets, labels, width=14):
    figure = plt.figure(figsize=(width, rows / cols * width))
    for i in range(cols * rows):
        dataset, label = datasets[i], labels[i]
        figure.add_subplot(rows, cols, i + 1)
        plt.title(label)
        plt.axis("off")
        plt.scatter(dataset[:, 0], dataset[:, 1], s=15, alpha=0.5)
    plt.savefig(trex_viz_path / f"scatter_{'_'.join(str(i) for i in labels)}.pdf")
    plt.show()
    plt.close()

# Visualize noising process
for i in range(len(datasaurus_series)):
    data = datasaurus_series[i]
    plt.figure(figsize=[6, 6])
    plt.scatter(data[:, 0], data[:, 1], s=15, alpha=0.5)
    plt.axis('off')
    plt.savefig(trex_viz_path / f"step{i:03d}.pdf")
    plt.close()

display_ts = [0, 6, 12, 25, 50]
scatters(len(display_ts), 1, [datasaurus_series[i] for i in display_ts], display_ts)

def drift_grids(dataset, xrange=(-2, 2, 0.15), yrange=(-2, 2, 0.15), n_steps=1000, beta_min=0.0001, beta_max=0.02):
    X, Y = np.meshgrid(np.arange(*xrange), np.arange(*yrange))
    XY = torch.tensor(np.stack((X, Y), axis=-1))
    beta = torch.linspace(beta_min, beta_max, n_steps)
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    dataset_grid = torch.tensor(np.full(XY.shape[:-1] + dataset.shape, dataset))
    s = torch.zeros_like(XY)
    series = [s.numpy()]  # Convert initial tensor to NumPy array
    for t in range(n_steps):
        alpha_bar_t1 = alpha_bar[t - 1] if t > 0 else torch.tensor(1.)
        xts = XY[:, :, None, :]  # [xlen, ylen, 1, 2]
        x0s = dataset_grid  # [xlen, ylen, N, 2]
        mus = (alpha[t].sqrt() * (1 - alpha_bar_t1) * xts + alpha_bar_t1.sqrt() * (1 - alpha[t]) * x0s) / (1 - alpha_bar[t])
        weights = torch.exp(-(torch.linalg.vector_norm(xts - alpha_bar[t].sqrt() * x0s, ord=2, dim=-1) / (1 - alpha_bar[t])) / 2)
        weights = weights / torch.sum(weights, dim=2, keepdim=True)
        mus = torch.sum(mus * weights[..., None], dim=2)
        drift = mus - xts.reshape(mus.shape)
        series.append(drift.numpy())
    return series

def quivers(cols, rows, datasets, labels, width=14, xrange=(-2, 2, 0.15), yrange=(-2, 2, 0.15)):
    X, Y = np.meshgrid(np.arange(*xrange), np.arange(*yrange))
    figure = plt.figure(figsize=(width, rows / cols * width))
    for i in range(cols * rows):
        dataset, label = datasets[i], labels[i]
        figure.add_subplot(rows, cols, i + 1)
        plt.title(label)
        plt.axis("off")
        plt.quiver(X, Y, dataset[:, :, 0], dataset[:, :, 1])
    plt.savefig(trex_viz_path / f"quiver_{'_'.join(str(i) for i in labels)}.pdf")
    plt.show()
    plt.close()

# Compute and visualize drift grids
xrange = (-2, 2, 0.15)
yrange = (-2, 2, 0.15)
datasaurus_drift = drift_grids(datasaurus_tensor, beta_min=(1 - 0.9999), beta_max=(1 - 0.95), n_steps=50, xrange=xrange, yrange=yrange)
quivers(len(display_ts), 1, [datasaurus_drift[i] for i in display_ts], display_ts, xrange=xrange, yrange=yrange)

for i in range(len(datasaurus_drift)):
    X, Y = np.meshgrid(np.arange(*xrange), np.arange(*yrange))
    data = datasaurus_drift[i]
    plt.figure(figsize=[6, 6])
    plt.quiver(X, Y, data[:, :, 0], data[:, :, 1])
    plt.axis('off')
    plt.savefig(trex_viz_path / f"drift{i:03d}.pdf")
    plt.close()

class DenoisingMLP(torch.nn.Module):
    def __init__(self, device, T, input_embedding='fourier', time_embedding='fourier'):
        super().__init__()
        self.T = T
        self.input_embedding = input_embedding
        if input_embedding == 'fourier':
            self.input_L = 64
            self.input_B = torch.randn((self.input_L // 2, 2)).to(device)
        elif input_embedding == 'identity':
            self.input_L = 2
        else:
            raise Exception("unknown input embedding")
        self.time_embedding = time_embedding
        if time_embedding == 'fourier':
            self.time_L = 32
            self.time_B = torch.randn((self.time_L // 2, 1)).to(device)
        elif time_embedding == 'linear':
            self.time_L = 1
        elif time_embedding == 'zero':
            self.time_L = 0
        else:
            raise Exception("unknown time embedding")
        nh = 64
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.input_L + self.time_L, nh),
            torch.nn.ReLU(),
            torch.nn.Linear(nh, nh),
            torch.nn.ReLU(),
            torch.nn.Linear(nh, nh),
            torch.nn.ReLU(),
            torch.nn.Linear(nh, nh),
            torch.nn.ReLU(),
            torch.nn.Linear(nh, 2),
        ).to(device)

    def forward(self, xt, t):
        if xt.shape[0] != t.shape[0]:
            raise Exception("expect t.shape[0]==xt.shape[0]")
        if self.input_embedding == 'fourier':
            xt = (self.input_B @ xt.T).T
            xt = torch.cat((torch.sin(xt), torch.cos(xt)), dim=1)
        if self.time_embedding == 'fourier':
            t = t / self.T - 0.5
            t = (self.time_B @ t[..., None].T).T
            t = torch.cat((torch.sin(t), torch.cos(t)), dim=1)
        elif self.time_embedding == 'linear':
            t = t[..., None] / self.T - 0.5
        x = torch.cat((xt, t.reshape(-1, self.time_L)), dim=1) if self.time_L > 0 else xt
        return self.layers(x)

# Updated training function to track both train and test losses
def train_with_loss_tracking(model, train_dataloader, test_dataloader, alpha_min=0.94, alpha_max=0.999, T=200, n_epochs=50, lr=4e-3, device='cpu'):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    alpha = torch.linspace(alpha_max, alpha_min, T, device=device)
    alpha_bar = torch.cumprod(alpha, dim=-1).reshape(-1, 1)
    train_losses, test_losses = [], []
    for e in range(n_epochs):
        model.train()
        batch_train_losses = []
        for x0s in train_dataloader:
            x0s = x0s[0]
            eps = torch.randn_like(x0s)
            t = torch.randint(T, (x0s.shape[0],), device=device)
            xts = alpha_bar[t].sqrt() * x0s + (1. - alpha_bar[t]).sqrt() * eps
            eps_pred = model(xts, t)
            loss = torch.nn.functional.mse_loss(eps_pred, eps)
            batch_train_losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
        train_losses.append(sum(batch_train_losses) / len(batch_train_losses))

        model.eval()
        with torch.no_grad():
            test_batch_losses = []
            test_batch_sizes = []
            for x0s in test_dataloader:
                x0s = x0s[0]
                eps = torch.randn_like(x0s)
                t = torch.randint(T, (x0s.shape[0],), device=device)
                xts = alpha_bar[t].sqrt() * x0s + (1. - alpha_bar[t]).sqrt() * eps
                eps_pred = model(xts, t)
                loss = torch.nn.functional.mse_loss(eps_pred, eps)
                test_batch_losses.append(loss.item() * x0s.shape[0])
                test_batch_sizes.append(x0s.shape[0])
            test_loss = sum(test_batch_losses) / sum(test_batch_sizes)
            test_losses.append(test_loss)
        if e % 100 == 0:
            print(f"Epoch {e}, Train Loss: {train_losses[-1]:.6f}, Test Loss: {test_losses[-1]:.6f}")
    plt.figure(figsize=[8, 6])
    plt.plot(np.arange(n_epochs), train_losses, label='Train Loss')
    plt.plot(np.arange(n_epochs), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(trex_viz_path / f"loss_curves_{model.input_embedding}_{model.time_embedding}.pdf")
    plt.show()
    plt.close()
    return train_losses, test_losses

# Noise prediction error analysis
def noise_prediction_error(model, dataloader, T, alpha_min=0.94, alpha_max=0.999, device='cpu'):
    alpha = torch.linspace(alpha_max, alpha_min, T).to(device)
    alpha_bar = torch.cumprod(alpha, dim=-1).reshape(-1, 1)
    errors = []
    model.eval()
    with torch.no_grad():
        for t in range(T):
            batch_errors = []
            for x0s in dataloader:
                x0s = x0s[0]
                eps = torch.randn_like(x0s)
                t_batch = torch.full((x0s.shape[0],), t).to(device)
                xts = alpha_bar[t].sqrt() * x0s + (1. - alpha_bar[t]).sqrt() * eps
                eps_pred = model(xts, t_batch)
                error = torch.nn.functional.mse_loss(eps_pred, eps).item()
                batch_errors.append(error)
            errors.append(sum(batch_errors) / len(batch_errors))
    model_name = f"{model.input_embedding}_{model.time_embedding}"
    plt.figure(figsize=[8, 6])
    plt.plot(np.arange(T), errors)
    plt.xlabel('Timestep')
    plt.ylabel('MSE')
    plt.title(f'Noise Prediction Error by Timestep ({model.input_embedding}, {model.time_embedding})')
    plt.savefig(trex_viz_path / f"noise_error_{model.input_embedding}_{model.time_embedding}.pdf")
    plt.show()
    plt.close()
    return errors

# Trajectory analysis functions
def plot_trajectories(steps, num_points=10, save_path=trex_viz_path / 'trajectories.pdf'):
    plt.figure(figsize=[8, 8])
    steps_array = torch.stack(steps).numpy()  # Shape: [T, num_samples, 2]
    indices = np.random.choice(steps_array.shape[1], num_points, replace=False)
    for idx in indices:
        trajectory = steps_array[:, idx, :]  # Shape: [T, 2]
        plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.5, marker='o', markersize=3)
    plt.axis('equal')
    plt.title('Point Trajectories During Denoising')
    plt.savefig(save_path)
    plt.show()
    plt.close()

def trajectory_displacement(steps,e1,e2):
    steps_array = torch.stack(steps).numpy()  # Shape: [T, num_samples, 2]
    displacements = np.sqrt(np.sum(np.diff(steps_array, axis=0) ** 2, axis=2))  # Shape: [T-1, num_samples]
    total_displacement = np.sum(displacements, axis=0)  # Shape: [num_samples]
    plt.figure(figsize=[8, 6])
    plt.hist(total_displacement, bins=50)
    plt.xlabel('Total Displacement')
    plt.ylabel('Frequency')
    plt.title('Distribution of Point Displacements')
    plt.savefig(trex_viz_path / 'displacement_histogram_{}_{}.pdf'.format(e1,e2))
    plt.show()
    plt.close()
    return total_displacement

def trajectory_velocity(steps,e1,e2):
    steps_array = torch.stack(steps).numpy()
    velocities = np.sqrt(np.sum(np.diff(steps_array, axis=0) ** 2, axis=2))  # Shape: [T-1, num_samples]
    mean_velocity = np.mean(velocities, axis=1)  # Average velocity per timestep
    plt.figure(figsize=[8, 6])
    plt.plot(np.arange(len(mean_velocity)), mean_velocity)
    plt.xlabel('Timestep')
    plt.ylabel('Mean Velocity')
    plt.title('Mean Velocity of Points Over Time')
    plt.savefig(trex_viz_path / 'velocity_{}_{}.pdf'.format(e1,e2))
    plt.show()
    plt.close()
    return mean_velocity

def cluster_trajectories(steps, num_clusters, e1, e2):
    steps_array = torch.stack(steps).numpy()  # Shape: [T, num_samples, 2]
    trajectories = steps_array.transpose(1, 0, 2).reshape(steps_array.shape[1], -1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(trajectories)
    labels = kmeans.labels_
    plt.figure(figsize=[8, 8])
    plt.scatter(steps[-1][:, 0], steps[-1][:, 1], c=labels, cmap='viridis', s=15, alpha=0.5)
    plt.axis('equal')
    plt.title('Clustered Trajectories in Final Point Cloud')
    plt.savefig(trex_viz_path / 'trajectory_clusters_{}_{}.pdf'.format(e1,e2))
    plt.show()
    plt.close()
    return labels

# Data distribution analysis
def compare_distributions(original, generated, save_path=trex_viz_path / 'distribution_comparison.pdf'):
    plt.figure(figsize=[12, 6])
    plt.subplot(1, 2, 1)
    plt.scatter(original[:, 0], original[:, 1], s=15, alpha=0.5)
    plt.title('Original Data')
    plt.axis('equal')
    plt.subplot(1, 2, 2)
    plt.scatter(generated[:, 0], generated[:, 1], s=15, alpha=0.5)
    plt.title('Generated Data')
    plt.axis('equal')
    plt.savefig(save_path)
    plt.show()
    plt.close()

def compute_wasserstein(original, generated):
    dist_x = wasserstein_distance(original[:, 0], generated[:, 0])
    dist_y = wasserstein_distance(original[:, 1], generated[:, 1])
    return (dist_x + dist_y) / 2

# Drift field analysis
def drift_magnitude(drift_series):
    magnitudes = [np.sqrt(np.sum(drift ** 2, axis=-1)) for drift in drift_series]
    plt.figure(figsize=[12, 6])
    for i, t in enumerate([1, 6, 12, 25, 50]):
        plt.subplot(1, 5, i + 1)
        plt.imshow(magnitudes[t], cmap='hot', origin='lower')
        plt.title(f'Timestep {t}')
        plt.axis('off')
    plt.colorbar(label='Drift Magnitude')
    plt.savefig(trex_viz_path / 'drift_magnitude.pdf')
    plt.show()
    plt.close()


def backward_drift_magnitude(steps, model, T, alpha_min=0.94, alpha_max=0.999, device='cpu', xrange=(-2, 2, 0.15), yrange=(-2, 2, 0.15)):
    X, Y = np.meshgrid(np.arange(*xrange), np.arange(*yrange))
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
    alpha = torch.linspace(alpha_max, alpha_min, T).to(device)
    alpha_bar = torch.cumprod(alpha, dim=-1).reshape(-1, 1)
    magnitudes = []
    
    model.eval()
    with torch.no_grad():
        for t in reversed(range(T)):
            t_batch = torch.full((grid_points_tensor.shape[0],), t).to(device)
            xt = grid_points_tensor
            noise_pred = model(xt, t_batch)
            mu_hat_t = (xt - (1 - alpha[t]) / (1 - alpha_bar[t]).sqrt() * noise_pred) / (alpha[t]).sqrt()
            drift = mu_hat_t - xt
            drift_magnitude = torch.sqrt(torch.sum(drift ** 2, dim=-1)).reshape(X.shape).cpu().numpy()
            magnitudes.append(drift_magnitude)
    
    # Plot the results
    plt.figure(figsize=[12, 6])
    display_ts = [0, 6, 12, 25, 49]  # Adjusted to match T=50 steps (0 to 49)
    for i, t in enumerate(display_ts):
        plt.subplot(1, 5, i + 1)
        plt.imshow(magnitudes[t], cmap='hot', origin='lower')
        plt.title(f'Timestep {49 - t}')  # Reverse timestep for backward process
        plt.axis('off')
    plt.colorbar(label='Backward Drift Magnitude')
    plt.savefig(trex_viz_path / 'backward_drift_magnitude.pdf')
    plt.show()
    plt.close()
from scipy.interpolate import griddata  # Ensure this is imported at the top of your script

def drift_direction(drift_series, steps, xrange=(-2, 2, 0.15), yrange=(-2, 2, 0.15)):
    from scipy.interpolate import griddata
    import numpy as np
    import matplotlib.pyplot as plt
    
    X, Y = np.meshgrid(np.arange(*xrange), np.arange(*yrange))
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
    final_points = steps[-1].numpy()  # Final state at timestep 0
    directions = []
    
    # Loop from 0 to 48 (states at t=49 to t=1), avoiding the final state
    for t in range(len(steps) - 1):  # t from 0 to 48
        points_t = steps[t].numpy()  # State at timestep 49 - t
        drift = drift_series[t + 1]  # Assuming drift_series[t+1] is drift for timestep 49 - t
        drift_flat = drift.reshape(-1, 2)
        
        # Interpolate drift at the current points
        drift_at_points = np.array([
            griddata(grid_points, drift_flat[:, 0], points_t, method='linear', fill_value=0),
            griddata(grid_points, drift_flat[:, 1], points_t, method='linear', fill_value=0)
        ]).T
        
        # Normalize drift
        drift_norm = np.sqrt(np.sum(drift_at_points ** 2, axis=-1, keepdims=True))
        drift_normalized = drift_at_points / (drift_norm + 1e-8)
        
        # Compute direction to final points
        direction_to_final = final_points - points_t
        direction_norm = np.sqrt(np.sum(direction_to_final ** 2, axis=-1, keepdims=True))
        direction_normalized = direction_to_final / (direction_norm + 1e-8)
        
        # Compute cosine similarity
        cosine_sim = np.sum(drift_normalized * direction_normalized, axis=-1)
        directions.append(np.mean(cosine_sim))
    
    # Plot the results
    plt.figure(figsize=[8, 6])
    plt.plot(np.arange(len(directions)), directions)
    plt.xlabel('Step')
    plt.ylabel('Mean Cosine Similarity')
    plt.title('Drift Direction Alignment with Final Points')
    plt.savefig(trex_viz_path / 'drift_direction.pdf')  # Adjust path as needed
    plt.show()
    plt.close()

@torch.no_grad()
def sample(num_samples, model, alpha_min=0.94, alpha_max=0.999, T=200, device='cpu'):
    alpha = torch.linspace(alpha_max, alpha_min, T).to(device)
    alpha_bar = torch.cumprod(alpha, dim=-1).reshape(-1, 1)
    steps = []
    xt = torch.randn((num_samples, 2)).to(device)
    for t in reversed(range(T)):
        t_batch = torch.full((num_samples,), t).to(device)
        noise_pred = model(xt, t_batch)
        mu_hat_t = (xt - (1 - alpha[t]) / (1 - alpha_bar[t]).sqrt() * noise_pred) / (alpha[t]).sqrt()
        z = torch.randn_like(xt).to(device)
        sigma = (1. - alpha[t]).sqrt()
        xt = mu_hat_t + sigma * z
        steps.append(xt.clone().detach().to('cpu'))

        # Save each step as SVG
        import matplotlib.pyplot as plt
        from pathlib import Path
        plt.figure(figsize=[6, 6])
        plt.scatter(xt[:, 0].cpu(), xt[:, 1].cpu(), s=15, alpha=0.5)
        plt.axis('off')
        plt.savefig(trex_viz_path / f"sample_step_{len(steps)-1:03d}.pdf")
        plt.close()
    return steps

# Data preparation
torch.manual_seed(42)
np.random.seed(42)
device = 'cpu'
idxs = torch.randperm(trex_viz_input.shape[0]).long()
num_train_data = int(idxs.shape[0] * 0.9)
train_data = torch.utils.data.TensorDataset(trex_viz_input[idxs[:num_train_data]].float().to(device))
test_data = torch.utils.data.TensorDataset(trex_viz_input[idxs[num_train_data:]].float().to(device))
bs = 32
train_dataloader = DataLoader(train_data, bs, shuffle=True)
test_dataloader = DataLoader(test_data, bs, shuffle=True)

# Train and analyze models
models = []
model_configs = [
    ('identity', 'zero', 0.95, 0.9999, 50),
    ('fourier', 'linear', 0.95, 0.9999, 50),
    ('fourier', 'fourier', 0.95, 0.9999, 50),
    ('fourier', 'fourier', 0.98, 0.9999, 50)
]
for input_emb, time_emb, alpha_min, alpha_max, T in model_configs:
    model = DenoisingMLP(device, T, input_emb, time_emb)
    train_losses, test_losses = train_with_loss_tracking(
        model, train_dataloader, test_dataloader, alpha_min=alpha_min, alpha_max=alpha_max, T=T, n_epochs=2000, lr=4e-4
    )
    errors = noise_prediction_error(model, test_dataloader, T, alpha_min, alpha_max)
    torch.manual_seed(42)
    steps = sample(1000, model, alpha_min=alpha_min, alpha_max=alpha_max, T=T)
    plt.figure(figsize=[6, 6])
    plt.scatter(steps[-1][:, 0], steps[-1][:, 1], s=15, alpha=0.5)
    plt.axis('off')
    plt.savefig(trex_viz_path / f"sample_{input_emb}_{time_emb}_{alpha_min}.pdf")
    plt.show()
    plt.close()

    # Trajectory analysis
    plot_trajectories(steps, num_points=10, save_path=trex_viz_path / f"trajectories_{input_emb}_{time_emb}.pdf")
    trajectory_displacement(steps,e1=input_emb,e2=time_emb)
    trajectory_velocity(steps,e1=input_emb,e2=time_emb)
    cluster_trajectories(steps, num_clusters=5,e1=input_emb,e2=time_emb)

    # Data distribution analysis
    compare_distributions(datasaurus_tensor, steps[-1], save_path=trex_viz_path / f"distribution_comparison_{input_emb}_{time_emb}.pdf")
    w_dist = compute_wasserstein(datasaurus_tensor.numpy(), steps[-1].numpy())
    print(f"Wasserstein Distance ({input_emb}, {time_emb}): {w_dist:.6f}")

    models.append(model)

# Drift field analysis (using the last model's steps)
drift_magnitude(datasaurus_drift)
drift_direction(datasaurus_drift, steps)
backward_drift_magnitude(steps, models[-1], T=50, alpha_min=0.95, alpha_max=0.9999, device='cpu', xrange=(-2, 2, 0.15), yrange=(-2, 2, 0.15))


# Compare trajectories across models
displacements = []
model_names = [f"{input_emb}_{time_emb}_{alpha_min}" for input_emb, time_emb, alpha_min, _, _ in model_configs]
for model, name, (input_emb, time_emb, alpha_min, alpha_max, T) in zip(models, model_names, model_configs):
    torch.manual_seed(42)
    steps = sample(1000, model, alpha_min=alpha_min, alpha_max=alpha_max, T=T)
    disp = trajectory_displacement(steps, e1=input_emb, e2=time_emb)
    displacements.append((name, disp))

plt.figure(figsize=[10, 6])
for name, disp in displacements:
    plt.hist(disp, bins=50, alpha=0.5, label=name)
plt.xlabel('Total Displacement')
plt.ylabel('Frequency')
plt.legend()
plt.title('Displacement Distribution Across Models')
plt.savefig(trex_viz_path / 'model_displacement_comparison.pdf')
plt.show()
plt.close()