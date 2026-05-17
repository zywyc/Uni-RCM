import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def gpu_kmeans(data, k, n_iter=20, batch_size=65536):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cpu = data.cpu() if data.is_cuda else data
    n, d = data_cpu.shape

    idx = torch.randperm(n)[:k]
    centroids = data_cpu[idx].to(device)

    final_counts = torch.zeros(k, device=device)

    for _ in range(n_iter):
        cluster_sums = torch.zeros(k, d, device=device)
        cluster_counts = torch.zeros(k, device=device)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = data_cpu[start:end].to(device)

            dist = (
                batch.pow(2).sum(1, keepdim=True)
                - 2 * batch @ centroids.t()
                + centroids.pow(2).sum(1)
            )
            assignments = dist.argmin(dim=1)

            enc = F.one_hot(assignments, k).float()
            cluster_sums += enc.t() @ batch
            cluster_counts += enc.sum(0)

        active = cluster_counts > 0
        centroids[active] = cluster_sums[active] / cluster_counts[active].unsqueeze(1)

        dead = ~active
        if dead.any():
            n_dead = dead.sum().item()
            reinit_idx = torch.randperm(n)[:n_dead]
            centroids[dead] = data_cpu[reinit_idx].to(device)

        final_counts = cluster_counts

    return centroids, final_counts


def build_quantizer(num_embeddings, embedding_dim, vq_layers=1):
    if vq_layers == 1:
        return VectorQuantizer(num_embeddings, embedding_dim)
    return ResidualVectorQuantizer(num_embeddings, embedding_dim, n_layers=vq_layers)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("initted", torch.tensor([False]))

    def quantize(self, z):
        """
        Nearest-neighbor lookup.

        Args:
            z: (..., D) input features

        Returns:
            z_q: nearest codebook vectors
            indices: codebook indices
            distances: L2 distance to nearest codebook vectors
        """
        need_reshape = False
        if z.dim() == 3:
            bsz, n_tokens, dim = z.shape
            z_flat = z.reshape(-1, dim)
            need_reshape = True
        else:
            z_flat = z

        dist = (z_flat.pow(2).sum(1, keepdim=True) - 2 * z_flat @ self.embed.t() + self.embed.pow(2).sum(1, keepdim=True).t())

        indices = dist.argmin(dim=1)
        z_q = self.embed[indices]
        distances = (z_flat - z_q).pow(2).sum(dim=1).sqrt()

        if need_reshape:
            z_q = z_q.reshape(bsz, n_tokens, dim)
            indices = indices.reshape(bsz, n_tokens)
            distances = distances.reshape(bsz, n_tokens)

        return z_q, indices, distances

    def get_utilization(self):
        active = (self.cluster_size > 1.0).sum().item()
        return active / self.num_embeddings

    def train_kmeans(self, data, n_iter=20, normalize=True):
        """
        Train codebook via K-Means.

        Args:
            data: (N, D) or (..., D)
            n_iter: iterations
            normalize: whether to L2-normalize before clustering
        """
        data_flat = data.reshape(-1, self.embedding_dim)

        if normalize:
            data_norm = F.normalize(data_flat, dim=-1)
            centroids, final_counts = gpu_kmeans(data_norm, self.num_embeddings, n_iter)

            device = centroids.device
            batch_sz = 65536
            centroid_sum = torch.zeros_like(centroids)
            centroid_cnt = torch.zeros(self.num_embeddings, device=device)
            n_flat = data_flat.shape[0]
            for start in range(0, n_flat, batch_sz):
                end = min(start + batch_sz, n_flat)
                batch_raw = data_flat[start:end].to(device)
                batch_n = F.normalize(batch_raw, dim=-1)
                d2c = (batch_n.pow(2).sum(1, keepdim=True) - 2 * batch_n @ centroids.t() + centroids.pow(2).sum(1))
                asn = d2c.argmin(dim=1)
                enc = F.one_hot(asn, self.num_embeddings).float()
                centroid_sum += enc.t() @ batch_raw
                centroid_cnt += enc.sum(0)

            active = centroid_cnt > 0
            final_centroids = centroids.clone()
            final_centroids[active] = centroid_sum[active] / centroid_cnt[active].unsqueeze(1)
            self.embed.data.copy_(final_centroids)
            self.cluster_size.data.copy_(centroid_cnt)
        else:
            centroids, final_counts = gpu_kmeans(data_flat, self.num_embeddings, n_iter)
            self.embed.data.copy_(centroids.to(self.embed.device))
            self.cluster_size.data.copy_(final_counts.to(self.cluster_size.device))

        self.embed_avg.data.copy_(self.embed.data)
        self.initted.fill_(True)


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, n_layers=4):
        super().__init__()
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.layers = nn.ModuleList(
            [VectorQuantizer(num_embeddings, embedding_dim) for _ in range(n_layers)]
        )

    def train_kmeans(self, data, n_iter=20, normalize=True):
        data_flat = data.reshape(-1, self.embedding_dim)
        device = self.layers[0].embed.device
        residual_cpu = data_flat.cpu() if data_flat.is_cuda else data_flat

        for i, layer in enumerate(self.layers):
            use_normalize = normalize if i == 0 else False
            layer.train_kmeans(residual_cpu, n_iter=n_iter, normalize=use_normalize)

            if i < self.n_layers - 1:
                new_residual = torch.empty_like(residual_cpu)
                batch_size = 65536
                for start in range(0, residual_cpu.shape[0], batch_size):
                    end = min(start + batch_size, residual_cpu.shape[0])
                    batch = residual_cpu[start:end].to(device)
                    z_q, _, _ = layer.quantize(batch)
                    new_residual[start:end] = (batch - z_q).cpu()
                residual_cpu = new_residual

    def quantize(self, z):
        need_reshape = False
        if z.dim() == 3:
            bsz, n_tokens, dim = z.shape
            z_flat = z.reshape(-1, dim)
            need_reshape = True
        else:
            z_flat = z

        residual = z_flat
        total_z_q = torch.zeros_like(z_flat)

        for layer in self.layers:
            z_q, _, _ = layer.quantize(residual)
            total_z_q = total_z_q + z_q
            residual = residual - z_q

        distances = residual.pow(2).sum(dim=-1).sqrt()

        if need_reshape:
            total_z_q = total_z_q.reshape(bsz, n_tokens, dim)
            distances = distances.reshape(bsz, n_tokens)

        return total_z_q, None, distances

    def get_utilization(self):
        utils = [layer.get_utilization() for layer in self.layers]
        return sum(utils) / len(utils)
