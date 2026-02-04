import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from typing import Sequence
from dataclasses import dataclass
from pathlib import Path

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    mutual_info_score,
    fowlkes_mallows_score,
    accuracy_score,
    f1_score,
    adjusted_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
)
from sklearn.cluster import KMeans

from model import VariationalAutoencoder
from dataloader import scDataset
from feature_optimization import BackFeature
from neighbor_clustering import NeighborClusteringEuclidean
import os
import sys
import importlib

def get_leiden_initial(X,resolution=0.6,seed=42):
    try:
        import scanpy as sc  # type: ignore
    except RuntimeError as e:
        msg = str(e)
        if "cannot cache function" in msg and os.environ.get("NUMBA_DISABLE_JIT") != "1":
            os.environ["NUMBA_DISABLE_JIT"] = "1"
            sys.modules.pop("scanpy", None)
            sc = importlib.import_module("scanpy")  # type: ignore
        else:
            raise

    adata = sc.AnnData(X)
    sc.pp.neighbors(adata, use_rep="X", random_state=seed)
    sc.tl.leiden(adata,resolution=resolution,random_state=seed)
    
    labels = adata.obs['leiden'].astype(int).values
    num_clusters = len(np.unique(labels))
    print(f'leiden initialization found {num_clusters} clusters (res = {resolution})')
    return labels,num_clusters

def build_vae_model(
    input_dim,z_dim,encode_layers,decode_layers,activation,sigma,alpha,gamma,device
):
    model = VariationalAutoencoder(
        input_dim=input_dim,
        z_dim=z_dim,
        encode_layers=encode_layers,
        decode_layers=decode_layers,
        activation=activation,
        sigma=sigma,
        alpha=alpha,
        gamma=gamma,
        device=device,
    )
    return model

def setup_seed(seed:int):
    """
    Set random seeds for reproducibility.
    """
    torch.manual_seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def get_rng_state() -> dict:
    """
    Capture RNG state for torch / numpy / python (and CUDA if available).
    """
    state = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state

def set_rng_state(state: dict | None) -> None:
    """
    Restore RNG state captured by get_rng_state.
    """
    if not state:
        return
    def _to_cpu_uint8_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.detach().to(device="cpu", dtype=torch.uint8)
        if isinstance(x, (bytes, bytearray)):
            return torch.tensor(list(x), dtype=torch.uint8, device="cpu")
        if isinstance(x, np.ndarray):
            return torch.tensor(x.astype(np.uint8, copy=False), dtype=torch.uint8, device="cpu")
        return torch.tensor(x, dtype=torch.uint8, device="cpu")

    if "torch_cpu" in state:
        try:
            torch.set_rng_state(_to_cpu_uint8_tensor(state["torch_cpu"]))
        except Exception as e:
            print(f"Warning: failed to restore torch CPU RNG state: {type(e).__name__}: {e}")
    if "torch_cuda" in state and torch.cuda.is_available():
        try:
            cuda_states = state["torch_cuda"]
            if isinstance(cuda_states, (list, tuple)):
                torch.cuda.set_rng_state_all([_to_cpu_uint8_tensor(s) for s in cuda_states])
            else:
                torch.cuda.set_rng_state_all([_to_cpu_uint8_tensor(cuda_states)])
        except Exception as e:
            print(f"Warning: failed to restore torch CUDA RNG state: {type(e).__name__}: {e}")


def _move_optimizer_state_to_device(optimizer: optim.Optimizer, device: torch.device) -> None:
    """
    Ensure optimizer state tensors live on the same device as model params.
    This matters when loading checkpoints with map_location='cpu'.
    """
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device=device)
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        random.setstate(state["python"])
    
def pretrain_vae(
    model,
    X,
    X_raw,
    size_factor,
    batch_size,
    optimizer,
    ae_save_path=None,
    epochs=400,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    dataset= TensorDataset(
        torch.tensor(X,dtype=torch.float32,device=device),
        torch.tensor(X_raw,dtype=torch.float32,device=device),
        torch.tensor(size_factor,dtype=torch.float32,device=device)
    )
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    print(f"Pretraining... Epochs:{epochs}")
    epoch_logs: list[dict] = []
    for epoch in range(epochs):
        loss_val = 0
        for _,(x_batch,x_raw_batch,size_factor_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            z0,mean,disp,pi = model.forward_AE(x_batch)
            recon_loss = model.zinb_loss(x_raw_batch,mean,disp,pi,size_factor_batch)
            loss = recon_loss
            loss.backward()
            optimizer.step()
            loss_val+=loss.item()*x_batch.size(0)
        avg_loss = float(loss_val / len(dataset))
        epoch_logs.append({"epoch": int(epoch + 1), "loss": avg_loss})
        print(f"Epoch{epoch+1}/{epochs}, Loss:{avg_loss:.4f}")
    print("pretraining finished")
    
    if ae_save_path:
        rng_state = get_rng_state()
        ae_save_path = Path(ae_save_path)
        ae_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss_val / len(dataset),
                # Save key hyperparameters to avoid accidental mismatch when reusing checkpoints.
                "pretrain_sigma": float(getattr(model, "sigma", float("nan"))),
                "input_dim": int(getattr(model, "input_dim", -1)),
                "z_dim": int(getattr(model, "z_dim", -1)),
                "encode_layers": list(getattr(model, "encode_layers", [])),
                "decode_layers": list(getattr(model, "decode_layers", [])),
                "activation": str(getattr(model, "activation", "")),
                "rng_state": rng_state,
            },
            ae_save_path
        )
        print(f"Pretrained model saved to {ae_save_path}")
    return epoch_logs


def parse_hidden_dims(config: str) -> Sequence[int]:
    dims = [int(dim) for dim in config.split(",") if dim.strip()]
    return dims


@dataclass
class SCDMCTrainConfig:
    adata_path: str
    raw_layer: str = "counts"
    label_key: str | None = 'Group'
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    z_dim: int =32
    encode_layers: str = "512,128"
    decode_layers: str = "128,512"
    activation: str = "relu"
    sigma: float = 1.0
    lr: float = 1e-4
    batch_size: int = 256
    pretrain_epochs: int = 700
    scdmc_epochs: int = 10
    max_rounds: int = 20
    min_clusters: int = 3
    lambda_feat: float = 0.2
    merge_lambda: float = 0.5
    disable_merge: bool = False
    init_method: str = "leiden"  # "leiden" (default), "kmeans", or "singleton"
    consensus_score_mode: str = "auto"  # "auto" (default), "raw", "complexity", "penalized"
    save_pretrain: bool = True
    pretrain_ckpt: str | None = None
    output_labels: str | None = None
    output_adata: str | None = None
    kmeans_k: int = 13
    resolution: float = 3
    
    def __post_init__(self):
        if self.save_pretrain and not self.pretrain_ckpt:
            repo_root = Path(__file__).resolve().parents[1]
            ckpt_name = f"{Path(self.adata_path).stem}_pretrain.pth.tar"
            self.pretrain_ckpt = str(repo_root / "pretrain" / ckpt_name)
        if self.output_adata is None:
            repo_root = Path(__file__).resolve().parents[1]
            self.output_adata = str(
                repo_root / "output" / f"{Path(self.adata_path).stem}_scDMC.h5ad"
            )

    @classmethod
    def default(cls) -> "SCDMCTrainConfig":
        repo_root = Path(__file__).resolve().parents[1]
        default_data = repo_root / "dataset" / "muraro.h5ad"
        output_labels = repo_root / "output" / f"{default_data.stem}_labels.csv"
        output_adata = repo_root / "output" / f"{default_data.stem}_scDMC.h5ad"
        return cls(
            adata_path=str(default_data),
            output_labels=str(output_labels),
            output_adata=str(output_adata),
        )


@torch.no_grad()
def compute_latent_views(model, data_tensor, batch_size, device):
    model.eval()
    num_cells = data_tensor.size(0)
    mus = []
    zs = []
    for start in range(0, num_cells, batch_size):
        end = min(start + batch_size, num_cells)
        x_batch = data_tensor[start:end].to(device)
        z_sample, mu, _ = model.encode_vae(x_batch)
        mus.append(mu.cpu())
        zs.append(z_sample.cpu())
    return torch.cat(mus, dim=0), torch.cat(zs, dim=0)
        
def train_back_feature_stage(
    model,
    data_tensor,
    raw_tensor,
    size_factors,
    pseudo_labels,
    batch_size,
    epochs,
    optimizer,
    back_feature_module,
    lambda_feat,
    device,
):
    model.train()
    num_cells = data_tensor.size(0)
    epoch_logs: list[dict] = []
    for epoch in range(epochs):
        perm = torch.randperm(num_cells)
        epoch_loss = 0.0
        epoch_zinb = 0.0
        epoch_feat = 0.0
        sample_count = 0
        for start in range(0, num_cells, batch_size):
            end = min(start + batch_size, num_cells)
            idx = perm[start:end]
            x_batch = data_tensor[idx].to(device)
            x_raw_batch = raw_tensor[idx].to(device)
            sf_batch = size_factors[idx].to(device)
            labels_batch = torch.tensor(
                pseudo_labels[idx.cpu().numpy()],
                dtype=torch.long,
                device=device,
            )

            optimizer.zero_grad()
            mu_batch, mean, disp, pi = model.forward_AE(x_batch)
            zinb_loss = model.zinb_loss(x_raw_batch, mean, disp, pi, sf_batch)
            feat_loss = back_feature_module.forward(mu_batch, labels_batch)
            loss = zinb_loss + lambda_feat * feat_loss
            loss.backward()
            optimizer.step()
            batch_size_curr = x_batch.size(0)
            epoch_loss += loss.item() * batch_size_curr
            epoch_zinb += zinb_loss.item() * batch_size_curr
            epoch_feat += feat_loss.item() * batch_size_curr
            sample_count += batch_size_curr
        avg_loss = epoch_loss / max(sample_count, 1)
        avg_zinb = epoch_zinb / max(sample_count, 1)
        avg_feat = epoch_feat / max(sample_count, 1)
        epoch_logs.append(
            {
                "epoch_in_round": int(epoch + 1),
                "loss_total": float(avg_loss),
                "loss_zinb": float(avg_zinb),
                "loss_feat": float(avg_feat),
            }
        )
        print(f"  BackFeature Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    return epoch_logs

def compute_consensus_score(labels1, labels2):
    """
    Compute two consensus scores between clusterings:
    - score_penalized: includes a trivial-solution penalty (recommended when K is large)
    - score_unpenalized: no trivial-solution penalty (recommended when K is small)
    """
    if isinstance(labels1, torch.Tensor):
        labels1 = labels1.cpu().numpy()
    if isinstance(labels2, torch.Tensor):
        labels2 = labels2.cpu().numpy()

    K1 = len(np.unique(labels1))
    K2 = len(np.unique(labels2))
    K = (K1 + K2) / 2.0
    N = len(labels1)

    # Trivial-solution penalty: discourages too few clusters.
    penalty_trivial = (K - 1.0) / (K + 1e-8)
    # Complexity penalty: discourages too many clusters.
    complexity_cost = (np.log(K) + 1e-8) / (np.log(N) + 1e-8)
    penalty_complexity = 1.0 - complexity_cost

    ami = adjusted_mutual_info_score(labels1, labels2)

    score_penalized = ami * penalty_complexity * penalty_trivial
    score_complexity = ami * penalty_complexity
    score_raw = ami

    return score_penalized, score_complexity, score_raw


def best_mapping(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    D = max(y_pred.max(), y_true.max()) + 1
    confusion = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        confusion[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(confusion.max() - confusion)
    total_correct = confusion[row_ind, col_ind].sum()
    return total_correct / len(y_pred)


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    acc = sum(w[i, j] for i, j in zip(row_ind, col_ind)) * 1.0 / y_pred.size
    return acc


def cluster_f1(y_true, y_pred) -> float:
    """
    Clustering F1 computed after optimally mapping cluster IDs to true labels.

    Important: raw cluster IDs are permutation-invariant, so we first align them
    to ground-truth labels via Hungarian matching, then compute macro-F1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_pred.size == y_true.size

    # Relabel to consecutive ints to avoid huge sparse matrices if labels are large.
    _, y_true_i = np.unique(y_true, return_inverse=True)
    _, y_pred_i = np.unique(y_pred, return_inverse=True)
    y_true_i = y_true_i.astype(np.int64)
    y_pred_i = y_pred_i.astype(np.int64)

    n_true = int(y_true_i.max()) + 1 if y_true_i.size else 0
    n_pred = int(y_pred_i.max()) + 1 if y_pred_i.size else 0
    if n_true == 0 or n_pred == 0:
        return 0.0

    w = np.zeros((n_pred, n_true), dtype=np.int64)
    np.add.at(w, (y_pred_i, y_true_i), 1)

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    mapping: dict[int, int] = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    # For any unmatched clusters (when n_pred != n_true), map to the majority true label.
    for r in range(n_pred):
        if r not in mapping:
            mapping[r] = int(np.argmax(w[r]))

    y_pred_mapped = np.array([mapping[int(c)] for c in y_pred_i], dtype=np.int64)
    return float(f1_score(y_true_i, y_pred_mapped, average="macro", zero_division=0))


def purity(y_true, y_pred):
    y_true = y_true.astype(np.int64).copy()
    y_pred = y_pred.astype(np.int64)
    y_voted_labels = np.zeros_like(y_true)
    labels = np.unique(y_true)
    label_mapping = {label: idx for idx, label in enumerate(labels)}
    y_true = np.array([label_mapping[label] for label in y_true])
    bins = np.arange(len(labels) + 1)
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner
    return accuracy_score(y_true, y_voted_labels)


def evaluate_metrics(y_true, y_pred, embedding=None):
    """
    Compute clustering metrics:
    - Agreement: MI / AMI / NMI / ARI / FMI
    - Label-based: ACC / PURITY / F1
    - Structure: Silhouette / Calinski-Harabasz (requires embedding)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics: dict[str, float | None] = {}
    metrics["mi"] = mutual_info_score(y_true, y_pred)
    metrics["ami"] = adjusted_mutual_info_score(y_true, y_pred)
    metrics["nmi"] = normalized_mutual_info_score(y_true, y_pred)
    metrics["ari"] = adjusted_rand_score(y_true, y_pred)
    metrics["fmi"] = fowlkes_mallows_score(y_true, y_pred)
    metrics["acc"] = cluster_acc(y_true, y_pred)
    metrics["purity"] = purity(y_true, y_pred)
    metrics["f1"] = cluster_f1(y_true, y_pred)

    # Structure metrics require embedding and at least two clusters.
    metrics["silhouette"] = None
    metrics["calinski_harabasz"] = None
    if embedding is not None:
        unique_labels = np.unique(y_pred)
        if unique_labels.size > 1 and unique_labels.size < len(y_pred):
            try:
                metrics["silhouette"] = float(silhouette_score(embedding, y_pred))
                metrics["calinski_harabasz"] = float(calinski_harabasz_score(embedding, y_pred))
            except Exception as e:
                print(f"Warning: failed to compute silhouette/CHI: {e}")
        else:
            print("Warning: silhouette/CHI skipped due to insufficient cluster diversity.")

    return metrics


def run_training(config: SCDMCTrainConfig):
    setup_seed(config.seed)

    device = torch.device(config.device)
    dataset = scDataset(
        adata_path=config.adata_path,
        raw_layer_name=config.raw_layer,
    )

    data_tensor = dataset.data
    raw_tensor = dataset.raw_data
    size_factor = dataset.size_factor
    num_cells, input_dim = data_tensor.shape
    gt_labels = None
    if config.label_key and config.label_key in dataset.adata.obs:
        raw_labels = np.asarray(dataset.adata.obs[config.label_key])
        _, gt_labels = np.unique(raw_labels, return_inverse=True)
        print(
            f"Found ground-truth labels in adata.obs['{config.label_key}'] "
            f"(K={int(np.unique(gt_labels).size)})."
        )
    else:
        print("Ground-truth labels not provided; metric computation disabled.")

    encode_layers = parse_hidden_dims(config.encode_layers)
    decode_layers = parse_hidden_dims(config.decode_layers)

    model = build_vae_model(
        input_dim=input_dim,
        z_dim=config.z_dim,
        encode_layers=encode_layers,
        decode_layers=decode_layers,
        activation=config.activation,
        sigma=config.sigma,
        alpha=1.0,
        gamma=0.0,
        device=device,
    )
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Stage 1: load a pretrained checkpoint if provided; otherwise pretrain.
    pretrain_epoch_logs: list[dict] | None = None
    if config.pretrain_ckpt is not None and Path(config.pretrain_ckpt).is_file():
        print(f"\n=== Stage 1: Loading pretrained ZINB-VAE from {config.pretrain_ckpt} ===")
        # Load on CPU to avoid accidentally moving RNG state tensors to CUDA.
        # Model weights will be copied onto the model's device by load_state_dict.
        ckpt = torch.load(config.pretrain_ckpt, map_location="cpu")
        ckpt_sigma = ckpt.get("pretrain_sigma", None)
        if ckpt_sigma is not None and abs(float(ckpt_sigma) - float(config.sigma)) > 1e-6:
            print(
                "Warning: pretrained checkpoint sigma mismatch "
                f"(ckpt={ckpt_sigma}, config={config.sigma}); will ignore checkpoint and re-pretrain."
            )
            ckpt = None
        if ckpt is not None:
            model.load_state_dict(ckpt["model_dict"])
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    _move_optimizer_state_to_device(optimizer, device)
                except Exception:
                    print("Warning: failed to load optimizer state, will re-init optimizer.")
            set_rng_state(ckpt.get("rng_state"))
        else:
            print("\n=== Stage 1: Pretraining ZINB-VAE ===")
            pretrain_epoch_logs = pretrain_vae(
                model,
                X=data_tensor.cpu().numpy(),
                X_raw=raw_tensor.cpu().numpy(),
                size_factor=size_factor.cpu().numpy(),
                batch_size=config.batch_size,
                optimizer=optimizer,
                ae_save_path=config.pretrain_ckpt,
                epochs=config.pretrain_epochs,
                device=str(device),
            )
    else:
        print("\n=== Stage 1: Pretraining ZINB-VAE ===")
        pretrain_epoch_logs = pretrain_vae(
            model,
            X=data_tensor.cpu().numpy(),
            X_raw=raw_tensor.cpu().numpy(),
            size_factor=size_factor.cpu().numpy(),
            batch_size=config.batch_size,
            optimizer=optimizer,
            ae_save_path=config.pretrain_ckpt,
            epochs=config.pretrain_epochs,
            device=str(device),
        )

    Z_mu_pre, _ = compute_latent_views(
        model,
        data_tensor,
        config.batch_size,
        device,
    )
    Z_mu_pre_np = Z_mu_pre.cpu().numpy()

    # Estimate K to pick the appropriate consensus score form.
    # NOTE: For clustering, labels are permutation-invariant. When init_method != "leiden",
    # we avoid using Leiden here as well to keep the ablation clean.
    init_method = str(config.init_method).lower()
    km_labels = None
    if init_method == "kmeans":
        kmeans = KMeans(
            n_clusters=config.kmeans_k,
            n_init=20,
            random_state=config.seed,
        )
        km_labels = kmeans.fit_predict(Z_mu_pre_np)

    if init_method == "leiden":
        _, K_eval = get_leiden_initial(
            Z_mu_pre_np,
            resolution=0.6,
            seed=config.seed,
        )
    elif init_method == "kmeans":
        K_eval = int(config.kmeans_k)
    elif init_method == "singleton":
        K_eval = int(num_cells)
    else:
        raise ValueError(f"Unknown init_method: {config.init_method}")
    consensus_mode = str(config.consensus_score_mode).lower()
    if consensus_mode not in {"auto", "raw", "complexity", "penalized"}:
        raise ValueError(f"Unknown consensus_score_mode: {config.consensus_score_mode}")

    use_penalized_score = K_eval > 6
    if consensus_mode == "auto":
        if use_penalized_score:
            print(
                f"Estimated K_eval={K_eval} (> 6); "
                "will use penalized consensus score (complexity + trivial penalties)."
            )
        else:
            print(
                f"Estimated K_eval={K_eval} (<= 6); "
                "will use complexity-only consensus score (no trivial penalty)."
            )
    elif consensus_mode == "raw":
        print("Consensus score mode: raw AMI (no penalty terms).")
    elif consensus_mode == "complexity":
        print("Consensus score mode: AMI with complexity penalty (no trivial penalty).")
    elif consensus_mode == "penalized":
        print("Consensus score mode: AMI with complexity + trivial penalties.")

    if init_method == "leiden":
        labels, num_clusters = get_leiden_initial(
            Z_mu_pre_np, resolution=config.resolution, seed=config.seed
        )
    elif init_method == "kmeans":
        if km_labels is None:
            raise RuntimeError("KMeans labels not computed; expected init_method='kmeans'.")
        labels = km_labels
        num_clusters = int(len(np.unique(labels)))
        print(f'kmeans initialization found {num_clusters} clusters (K = {num_clusters})')
    elif init_method == "singleton":
        labels = np.arange(num_cells, dtype=np.int64)
        num_clusters = int(num_cells)
        print(f"singleton initialization: start with K = N = {num_clusters} clusters")
    else:
        raise ValueError(f"Unknown init_method: {config.init_method}")
    
    clusterer = NeighborClusteringEuclidean(
        device=device,
        merge_lambda=float(config.merge_lambda),
        disable_merge=bool(config.disable_merge),
    )
    back_feature_module = BackFeature(
        batch_size=config.batch_size,
        num_classes=num_clusters,
        device=device
    )

    labels_mu = labels.copy()
    labels_sample = labels.copy()
    
    best_record = {
        'score':-1.0,
        'round':0,
        'K':0,
        'labels':None,
    }
    round_logs: list[dict] = []
    # Store per-round labels to select the best round by the global score sequence.
    round_labels: dict[int, np.ndarray] = {}
    stage2_epoch_logs: list[dict] = []
    stage2_global_epoch = 0
    
    print("\n=== Stage 2: SCDMC Self-Supervised Training ===")
    for round_idx in range(1, config.max_rounds + 1):
        print(f"\n--- Round {round_idx}/{config.max_rounds} ---")
        Z_mu, Z_sample = compute_latent_views(
            model,
            data_tensor,
            config.batch_size,
            device,
        )
        Z_mu_device = Z_mu.to(device)
        Z_sample_device = Z_sample.to(device)

        K_mu, labels_mu = clusterer.step(Z_mu_device, labels_mu)
        K_sample, labels_sample = clusterer.step(Z_sample_device, labels_sample)
        print(f"K_mu={K_mu}, K_sample={K_sample}")

        metrics = None
        if gt_labels is not None:
            metrics = evaluate_metrics(gt_labels, labels_mu)
            print(
                f"NMI: {metrics['nmi']:.4f}, ARI: {metrics['ari']:.4f}"
            )
        else:
            print("Ground-truth labels not provided; skipping per-round metric computation.")
        
        if K_mu < config.min_clusters or K_sample < config.min_clusters:
            print("Cluster number dropped below threshold; stopping iterations.")
            break

        score_penalized, score_complexity, score_raw = compute_consensus_score(labels_mu, labels_sample)
        if consensus_mode == "raw":
            score = float(score_raw)
            score_name = "raw"
        elif consensus_mode == "complexity":
            score = float(score_complexity)
            score_name = "complexity"
        elif consensus_mode == "penalized":
            score = float(score_penalized)
            score_name = "penalized"
        else:
            # auto
            if use_penalized_score:
                score = float(score_penalized)
                score_name = "penalized"
            else:
                score = float(score_complexity)
                score_name = "complexity"

        score_2dp = float(f"{score:.2f}")
        print(f"Consensus score ({score_name}) between views: {score_2dp:.2f}")

        round_log = {
            "round": int(round_idx),
            "K_mu": int(K_mu),
            "K_sample": int(K_sample),
            "consensus_score": score_2dp,
        }
        if metrics is not None:
            round_log["nmi"] = float(metrics["nmi"])
            round_log["ari"] = float(metrics["ari"])
        round_logs.append(round_log)

        if isinstance(labels_mu, torch.Tensor):
            round_labels[round_idx] = labels_mu.detach().cpu().numpy().copy()
        else:
            round_labels[round_idx] = np.asarray(labels_mu).copy()

        back_feature_module.num_classes = K_mu

        bf_epoch_logs = train_back_feature_stage(
            model=model,
            data_tensor=data_tensor,
            raw_tensor=raw_tensor,
            size_factors=size_factor,
            pseudo_labels=labels_mu,
            batch_size=config.batch_size,
            epochs=config.scdmc_epochs,
            optimizer=optimizer,
            back_feature_module=back_feature_module,
            lambda_feat=config.lambda_feat,
            device=device,
        )
        for item in bf_epoch_logs:
            stage2_global_epoch += 1
            stage2_epoch_logs.append(
                {
                    "epoch": int(stage2_global_epoch),
                    "round": int(round_idx),
                    **item,
                }
            )

    # Choose the final round by maximizing the consensus score (ties broken by earlier round).
    best_labels = labels_mu
    if round_logs:
        log_by_round = {log["round"]: log for log in round_logs}
        score_by_round = {
            r: float(log["consensus_score"]) for r, log in log_by_round.items()
        }
        all_rounds = sorted(score_by_round.keys())

        score100_by_round = {r: int(round(score_by_round[r] * 100)) for r in all_rounds}
        chosen_round = max(
            all_rounds,
            key=lambda r: (score100_by_round[r], -r),
        )

        chosen_score = score_by_round[chosen_round]

        if chosen_round in round_labels:
            best_labels = round_labels[chosen_round]
        best_record["round"] = int(chosen_round)
        best_record["score"] = float(chosen_score)

        chosen_K = 0
        for log in round_logs:
            if log["round"] == chosen_round:
                chosen_K = int(log["K_mu"])
                break
        best_record["K"] = chosen_K
        best_record["labels"] = best_labels

    if isinstance(best_labels, torch.Tensor):
        best_labels = best_labels.detach().cpu().numpy()
    else:
        best_labels = np.asarray(best_labels)
    best_labels = best_labels.astype(int)
    best_record["labels"] = best_labels

    for log in round_logs:
        log["is_selected"] = log["round"] == best_record["round"]

    print("\n=== Training Complete ===")
    print(f"Best consensus score: {best_record['score']:.2f}, Estimated K: {best_record['K']}, Round: {best_record['round']}")

    # Save final embedding and labels to adata.
    Z_mu_final, _ = compute_latent_views(
        model,
        data_tensor,
        config.batch_size,
        device,
    )
    final_embedding = Z_mu_final.cpu().numpy()
    final_labels = np.asarray(best_labels).astype(int)
    dataset.adata.obsm["X_scDMC"] = final_embedding
    dataset.adata.obs["scDMC_label"] = final_labels

    if config.output_adata:
        adata_out = Path(config.output_adata)
        adata_out.parent.mkdir(parents=True, exist_ok=True)
        dataset.adata.write_h5ad(adata_out)
        print(f"Annotated adata saved to {adata_out}")

    final_metrics = None
    metrics_str = None
    if gt_labels is not None:
        final_metrics = evaluate_metrics(gt_labels, best_labels, embedding=final_embedding)
        metrics_str = (
            "Evaluation Metrics - "
            f"MI: {final_metrics['mi']:.4f}, AMI: {final_metrics['ami']:.4f}, "
            f"NMI: {final_metrics['nmi']:.4f}, ARI: {final_metrics['ari']:.4f}, FMI: {final_metrics['fmi']:.4f}, "
            f"ACC: {final_metrics['acc']:.4f}, PURITY: {final_metrics['purity']:.4f}, F1: {final_metrics['f1']:.4f}"
        )

        if final_metrics.get("silhouette") is not None:
            metrics_str += f", SC: {final_metrics['silhouette']:.4f}"
        if final_metrics.get("calinski_harabasz") is not None:
            metrics_str += f", CHI: {final_metrics['calinski_harabasz']:.2f}"
        print(metrics_str)

    output_path = None
    if config.output_labels:
        output_path = Path(config.output_labels)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_path, best_labels, fmt="%d")
        print(f"Predicted labels saved to {output_path}")

    if metrics_str is not None:
        if output_path is not None:
            metrics_path = output_path.with_name(output_path.stem + "_metrics.txt")
        else:
            metrics_path = Path("scdmc_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(metrics_str)

    return {
        "round_logs": round_logs,
        "best_record": best_record,
        "final_metrics": final_metrics,
        "loss_logs": {
            "pretrain": pretrain_epoch_logs,
            "stage2": stage2_epoch_logs,
        },
    }


def main():
    config = SCDMCTrainConfig.default()
    setup_seed(config.seed)
    run_training(config)


if __name__ == "__main__":
    main()
