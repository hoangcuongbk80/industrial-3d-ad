import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def farthest_point_sample(xyz, n_point, is_center=False):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, n_point, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    if is_center:
        centroid = xyz.mean(1, keepdim=True)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.argmax(distance, dim=-1)
    else:
        farthest = torch.randint(0, N, (B,), device=device)
    batch_idx = torch.arange(B, device=device)
    for i in range(n_point):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, dim=-1)
    return centroids


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx]


def log_boltzmann_kernel(cost, u, v, epsilon):
    return (-cost + u.unsqueeze(-1) + v.unsqueeze(-2)) / epsilon


def sinkhorn_rpm(log_alpha, n_iters=20, slack=True, eps=-1):
    B, J, K = log_alpha.shape
    if slack:
        pad = nn.ZeroPad2d((0,1,0,1))
        log_p = pad(log_alpha.unsqueeze(1)).squeeze(1)
        for _ in range(n_iters):
            # row
            log_p = torch.cat([log_p[:, :-1] - torch.logsumexp(log_p[:, :-1], dim=2, keepdim=True),
                                log_p[:, -1:].clone()], dim=1)
            # col
            log_p = torch.cat([log_p[:, :, :-1] - torch.logsumexp(log_p[:, :, :-1], dim=1, keepdim=True),
                                log_p[:, :, -1:].clone()], dim=2)
        return log_p[:, :-1, :-1]
    else:
        for _ in range(n_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        return log_alpha


def query_ball_point(radius, nsample, xyz, new_xyz, self_idx=None):
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    sqrd = torch.cdist(new_xyz, xyz)
    idx = torch.arange(N, device=xyz.device).view(1,1,N).repeat(B, S, 1)
    if self_idx is not None:
        idx[torch.arange(B)[:,None], torch.arange(S)[None], self_idx] = N
    idx[sqrd > radius**2] = N
    idx = torch.sort(idx, dim=-1)[0][:,:,:nsample]
    first = (self_idx.unsqueeze(-1).repeat(1,1,nsample) if self_idx is not None
             else idx[:,:,0:1].repeat(1,1,nsample))
    mask = idx==N
    idx[mask] = first[mask]
    return idx


def index_gather(points, idx):
    B, N, C = points.shape
    B, S, K = idx.shape
    batch_indices = idx.unsqueeze(-1).expand(B, S, K, C)
    points_expand = points.unsqueeze(1).expand(B, S, N, C)
    return torch.gather(points_expand, 2, batch_indices)


def calculate_curvature_pca_ball(queries, refs, num_neighbors=10, radius=0.1, eps=1e-8):
    idx = query_ball_point(radius, num_neighbors, refs, queries)
    # include centroid
    mean_node = queries.mean(dim=1, keepdim=True)
    cat = torch.cat([refs, mean_node], dim=1)
    neighbor = index_gather(cat, idx)
    centered = neighbor - queries.unsqueeze(2)
    cov = centered.transpose(-2,-1) @ centered / num_neighbors
    eig = torch.linalg.eigvalsh(cov + eps)
    l2,l1,l0 = eig[:,:,2].clamp(min=eps), eig[:,:,1], eig[:,:,0]
    f1 = (l2-l1)/l2
    f2 = (l1-l0)/l2
    f3 = l0/l2
    return torch.stack([f1,f2,f3], dim=-1)


def node_to_group(node, xyz, feats, radius, n_sample, os=None, is_knn=True):
    B,N,_ = xyz.shape
    if os is None:
        os = torch.ones((B,N), device=xyz.device)/N
    center_xyz = (xyz*os.unsqueeze(-1)).sum(1, keepdim=True)
    center_feat = (feats*os.unsqueeze(-1)).sum(1, keepdim=True)
    idx = query_ball_point(radius, n_sample, xyz, node) if not is_knn else torch.topk(torch.cdist(node, xyz), n_sample, dim=-1, largest=False)[1]
    cat_xyz = torch.cat([xyz, center_xyz], dim=1)
    cat_feat = torch.cat([feats, center_feat], dim=1)
    grouped_xyz = index_gather(cat_xyz, idx)
    grouped_feat = index_gather(cat_feat, idx)
    return grouped_xyz, grouped_feat, idx


def get_prototype(features, similarity, top_k=20):
    sim_t = similarity.transpose(-1,-2)
    vals, ids = torch.topk(sim_t, top_k, dim=-1)
    vals = vals/vals.sum(-1,True).clamp(min=1e-4)
    ids_exp = ids.unsqueeze(-1).expand(-1,-1,-1,features.size(-1))
    patches = torch.gather(features.unsqueeze(1).expand(-1,sim_t.size(1),-1,-1),2,ids_exp)
    protos = (patches * vals.unsqueeze(-1)).sum(2)
    return protos

def sinkhorn_transport(cost, tau=0.05, iters=20):
    log_alpha = -cost/tau
    log_gamma = sinkhorn_rpm(log_alpha, n_iters=iters)
    return torch.exp(log_gamma)

class SpatialContextAggregation(nn.Module):
    def __init__(self, num_prototypes=100, K=20, tau=0.05, radius=0.2):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.K = K
        self.tau = tau
        self.radius = radius

    def forward(self, xyz, feats):
        B,N,C = feats.shape
        # 1) downsample prototypes via FPS
        proto_idx = farthest_point_sample(xyz, self.num_prototypes)
        p_xyz = index_points(xyz, proto_idx)
        p_feat = index_points(feats, proto_idx)
        # 2) local neighborhood/geometric features
        geom = calculate_curvature_pca_ball(p_xyz.view(-1,self.num_prototypes,3), xyz)
        # 3) compute cost matrix
        dist_sp = torch.cdist(xyz, p_xyz)/ (C**0.5)
        dist_ge = torch.norm(geom,dim=-1)/ (geom.size(-1)**0.5)
        cost = dist_sp + dist_ge
        # 4) transport
        gamma = sinkhorn_transport(cost, tau=self.tau)
        # 5) update prototypes
        p_feat = 0.5*p_feat + 0.5*(gamma.unsqueeze(-1)*feats.unsqueeze(1)).sum(2)
        # 6) prototype fusion via similarity
        sim = torch.einsum('bmc,bkc->bmk', p_feat, p_feat)/C**0.5
        agg = get_prototype(p_feat, sim, top_k=self.K)
        p_feat = 0.5*p_feat + 0.5*agg
        # 7) backpropagate to points
        sim_pt = torch.einsum('bnc,bmc->bnm', feats, p_feat)/C**0.5
        w = sinkhorn_transport(sim_pt, tau=self.tau)
        refined = 0.5*feats + 0.5*(w.unsqueeze(-1)*p_feat.unsqueeze(1)).sum(2)
        return refined

class FeatureAdaptor(nn.Module):
    def __init__(self, C=256, bottleneck=128, alpha=0.5):
        super().__init__()
        self.fc1 = nn.Linear(C, bottleneck)
        self.bn1 = nn.BatchNorm1d(bottleneck)
        self.fc2 = nn.Linear(bottleneck, C)
        self.bn2 = nn.BatchNorm1d(C)
        self.alpha = alpha

    def forward(self, x):
        B,N,C = x.shape
        h = self.bn1(self.fc1(x.view(-1,C)))
        u = F.leaky_relu(h,0.1)
        q = self.bn2(self.fc2(u)).view(B,N,C)
        return x + self.alpha*(q-x)

class AnomalousFeatureGenerator(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma
    def forward(self, x):
        return x + torch.randn_like(x)*self.sigma

class Discriminator(nn.Module):
    def __init__(self, C=256):
        super().__init__()
        self.fc1 = nn.Linear(C, C)
        self.bn1 = nn.BatchNorm1d(C)
        self.act = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(C, 1)

    def forward(self, x):
        B,N,C = x.shape
        h = self.bn1(self.fc1(x.view(-1,C)))
        h = self.act(h)
        out = self.fc2(h)
        return out.view(B,N)

class Industrial3DAnomalyDetector(nn.Module):
    def __init__(self, backbone, sca_cfg={}, adapt_cfg={}, sigma=0.1):
        super().__init__()
        self.backbone = backbone
        self.sca = SpatialContextAggregation(**sca_cfg)
        self.adaptor = FeatureAdaptor(**adapt_cfg)
        self.gen = AnomalousFeatureGenerator(sigma)
        self.disc = Discriminator(adapt_cfg.get('C',256))

    def forward(self, inputs, training=False):
        xyz, feats = self.backbone(inputs)
        refined = self.sca(xyz, feats)
        adapted = self.adaptor(refined)
        scores = self.disc(adapted)
        if training:
            perturbed = self.gen(adapted)
            return scores, adapted, perturbed
        return -scores
