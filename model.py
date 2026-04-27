import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from tqdm import tqdm

def trans_to_cuda(variable):
    return variable.cuda() if torch.cuda.is_available() else variable

def trans_to_cpu(variable):
    return variable.cpu() if torch.cuda.is_available() else variable


def safe_sparse_mm(adjacency, dense):
    """
    Sparse-dense matmul with stable dtype boundaries:
    sparse matmul runs in float32, output is cast back to dense dtype.
    """
    out = torch.sparse.mm(adjacency, dense.float())
    return out.to(dense.dtype) if out.dtype != dense.dtype else out


class ItemConv(Module):
    def __init__(self, layers, K1, K2, K3, dropout, alpha, emb_size=100):
        super(ItemConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.w_item = nn.ModuleDict()
        self.w_i1 = nn.ModuleDict()
        self.w_i2 = nn.ModuleDict()
        self.dropout = nn.Dropout(p=dropout)
        self.k1 = K1
        self.k2 = K2
        self.k3 = K3
        self.channel = 3
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(alpha)

        for i in range(self.channel):
            self.w_i1[f'weight_item{i}'] = nn.Linear(self.emb_size, self.emb_size, bias=False)

        for i in range(self.layers):
            self.w_item[f'weight_item{i}'] = nn.Linear(self.emb_size, self.emb_size, bias=False)

        self.w_i2['weight_item0'] = nn.Linear(self.emb_size, self.k1, bias=False)
        self.w_i2['weight_item1'] = nn.Linear(self.emb_size, self.k2, bias=False)
        self.w_i2['weight_item2'] = nn.Linear(self.emb_size, self.k3, bias=False)

    def forward(self, adj, adjacency, embedding, channel):
        item_embeddings = embedding
        final = [item_embeddings]
        finalh = []
        for i in range(self.layers):
            item_embeddings = self.w_item[f'weight_item{i}'](item_embeddings)
            item_embeddings = safe_sparse_mm(adjacency, item_embeddings)

            H1 = self.w_i1[f'weight_item{channel}'](item_embeddings) + item_embeddings
            H1 = self.relu(H1)
            H1 = self.w_i2[f'weight_item{channel}'](H1)
            H1 = torch.softmax(H1, dim=1)

            h = H1.T.mul(adj)
            h = h.mul(1.0 / (torch.sum(h, dim=0) + 1e-8))
            h = h @ item_embeddings
            h = H1 @ h
            item_embeddings = h + item_embeddings
            final.append(F.normalize(item_embeddings, dim=-1, p=2))
            finalh.append(F.normalize(h, dim=-1, p=2))

        item_embeddings = torch.sum(torch.stack(final), 0) / (self.layers + 1)
        hs = torch.sum(torch.stack(finalh), 0) / max(self.layers, 1)
        return item_embeddings, hs

class HyperGraphConv(Module):
    """Lightweight hypergraph convolution on item nodes using a precomputed sparse propagation matrix."""
    def __init__(self, layers, dropout, emb_size=100):
        super(HyperGraphConv, self).__init__()
        self.layers = layers
        self.emb_size = emb_size
        self.dropout = nn.Dropout(p=dropout)
        self.linears = nn.ModuleList([nn.Linear(self.emb_size, self.emb_size, bias=False) for _ in range(self.layers)])

    def forward(self, hyper_propagation, embedding):
        """Args: hyper_propagation [N,N] sparse tensor, embedding [N,D]; Returns: refined embedding [N,D]."""
        x = embedding
        outs = [F.normalize(x, dim=-1, p=2)]
        for l in range(self.layers):
            x = self.linears[l](x)
            x = safe_sparse_mm(hyper_propagation, x)
            x = F.relu(x)
            x = self.dropout(x)
            outs.append(F.normalize(x, dim=-1, p=2))
        return torch.sum(torch.stack(outs), 0) / len(outs)


class MDHG(Module):
    def __init__(self, R, adj1, adj2, adjacency, adjacency_T, adjacency1, adjacency_comp, adjacency_sub, hyper_comp, hyper_sub, R1, comp_deg, sub_deg,
                 adjacency_fuzzy, adjacency_T_fuzzy, adjacency1_fuzzy,
                 adj1_fuzzy, adj2_fuzzy, R_fuzzy, R1_fuzzy,
                 n_node, lr, layers, l2, beta, lam, eps, dataset,
                 K1, K2, K3, dropout, alpha, emb_size=100, batch_size=100,
                 intent_align_weight=0.03, short_intent_min=0.10, short_intent_max=0.45,
                 short_len_factor_min=0.35, comp_sub_pair_hyper_mix=0.5, comp_sub_decouple_weight=0.02,
                 logit_comp_scale=0.20, logit_sub_scale=0.25, logit_short_sub_boost=0.30,
                 comp_sub_warmup_epochs=1, comp_sub_ramp_epochs=4,
                 rel_conf_comp_scale=1.0, rel_conf_sub_scale=1.0,
                 rel_conf_event_gain=0.20, rel_conf_repeat_penalty=0.25, rel_conf_len_gain=0.15,
                 enable_comp_branch=True, enable_sub_branch=True,
                 enable_logit_residual=True, enable_rel_conf_gate=True):
        super(MDHG, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.dataset = dataset
        self.lr = lr
        self.layers = layers
        self.use_amp = False
        self.w_k = 10
        self.numerical_eps = 1e-8
        self.adjacency = trans_to_cuda(self.trans_adj(adjacency))
        self.adjacency_T = trans_to_cuda(self.trans_adj(adjacency_T))
        self.adjacency1 = trans_to_cuda(self.trans_adj(adjacency1))
        self.adjacency_comp = trans_to_cuda(self.trans_adj(adjacency_comp))
        self.adjacency_sub = trans_to_cuda(self.trans_adj(adjacency_sub))
        self.hyper_comp = trans_to_cuda(self.trans_adj(hyper_comp))
        self.hyper_sub = trans_to_cuda(self.trans_adj(hyper_sub))
        self.adjacency_fuzzy = trans_to_cuda(self.trans_adj(adjacency_fuzzy))
        self.adjacency_T_fuzzy = trans_to_cuda(self.trans_adj(adjacency_T_fuzzy))
        self.adjacency1_fuzzy = trans_to_cuda(self.trans_adj(adjacency1_fuzzy))
        self.adj1 = torch.cuda.FloatTensor(adj1) if torch.cuda.is_available() else torch.FloatTensor(adj1)
        self.adj2 = torch.cuda.FloatTensor(adj2) if torch.cuda.is_available() else torch.FloatTensor(adj2)
        self.R1 = torch.cuda.FloatTensor(R1) if torch.cuda.is_available() else torch.FloatTensor(R1)
        self.adj1_fuzzy = torch.cuda.FloatTensor(adj1_fuzzy) if torch.cuda.is_available() else torch.FloatTensor(adj1_fuzzy)
        self.adj2_fuzzy = torch.cuda.FloatTensor(adj2_fuzzy) if torch.cuda.is_available() else torch.FloatTensor(adj2_fuzzy)
        self.R_fuzzy = torch.cuda.FloatTensor(R_fuzzy) if torch.cuda.is_available() else torch.FloatTensor(R_fuzzy)
        self.comp_deg = torch.cuda.FloatTensor(comp_deg) if torch.cuda.is_available() else torch.FloatTensor(comp_deg)
        self.sub_deg = torch.cuda.FloatTensor(sub_deg) if torch.cuda.is_available() else torch.FloatTensor(sub_deg)
        self.R_fuzzy = self.R_fuzzy.reshape(-1)
        if self.R_fuzzy.numel() < self.n_node:
            pad = torch.ones(self.n_node - self.R_fuzzy.numel(), device=self.R_fuzzy.device)
            self.R_fuzzy = torch.cat([self.R_fuzzy, pad], dim=0)
        elif self.R_fuzzy.numel() > self.n_node:
            self.R_fuzzy = self.R_fuzzy[:self.n_node]
        self.R1_fuzzy = torch.cuda.FloatTensor(R1_fuzzy) if torch.cuda.is_available() else torch.FloatTensor(R1_fuzzy)
        self.embedding1 = nn.Embedding(self.n_node, self.emb_size)
        self.embedding2 = nn.Embedding(self.n_node, self.emb_size)
        self.embedding3 = nn.Embedding(self.n_node, self.emb_size)
        self.event_embedding = nn.Embedding(4, self.emb_size)
        self.event_scale = nn.Embedding(4, 1)
        self.pos_decay = 0.2
        self.pos_len = 1000 if self.dataset == 'retailrocket' else 200
        self.pos_embedding = nn.Embedding(self.pos_len, self.emb_size)
        self.ItemGraph = ItemConv(layers, K1, K2, K3, dropout, alpha, emb_size=self.emb_size)
        self.CompHyperGraph = HyperGraphConv(layers, dropout, emb_size=self.emb_size)
        self.SubHyperGraph = HyperGraphConv(layers, dropout, emb_size=self.emb_size)
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.glu1 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=True)

        self.attention = nn.Parameter(torch.Tensor(1, self.emb_size))
        self.attention_mat = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size))

        self.gate_mlp = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, 3)
        )
        self.gate_dropout = nn.Dropout(p=0.05)
        self.final_dropout = nn.Dropout(p=0.1)
        self.short_intent_mlp = nn.Sequential(
            nn.Linear(2 * self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, 1)
        )

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # ---- 稳定版损失配置（先保主任务）----
        self.fuzzy_prior_strength = 0.35
        self.rank_margin = 0.30
        self.rel_loss_weight = 0.05
        self.sess_loss_weight = 0.01
        self.rank_loss_weight = 0.18
        self.hyper_loss_weight = 0.01
        self.soft_loss_weight = 0.02
        self.warmup_epochs = 0
        self.max_fuzzy_factor = 0.25
        self.hyperedge_min_prob = 0.35
        self.hyperedge_event_gain = 0.20
        self.hyperedge_repeat_penalty = 0.30
        self.item_prior_mix = 0.10
        self.soft_label_smooth_min = 0.02
        self.soft_label_smooth_gain = 0.08
        self.soft_label_smooth_max = 0.06
        self.label_smoothing = 0.06
        self.bpr_loss_weight = 0.18
        self.intent_align_weight = intent_align_weight
        self.short_intent_min = short_intent_min
        self.short_intent_max = short_intent_max
        self.short_len_factor_min = short_len_factor_min
        self.topk_hardneg = 100
        self.score_temperature = 0.85

        # item-view complementary/substitute fusion weights
        self.comp_weight_base = 0.60
        self.comp_weight_decay = 0.35
        self.sub_weight_base = 0.25
        self.sub_weight_gain = 0.45
        self.comp_weight_min = 0.15
        self.comp_weight_max = 0.70
        self.sub_weight_min = 0.15
        self.sub_weight_max = 0.75
        self.base_weight_min = 0.10
        self.base_weight_max = 0.70
        self.comp_sub_pair_hyper_mix = comp_sub_pair_hyper_mix
        self.comp_sub_decouple_weight = comp_sub_decouple_weight
        self.logit_comp_scale = logit_comp_scale
        self.logit_sub_scale = logit_sub_scale
        self.logit_short_sub_boost = logit_short_sub_boost
        self.sub_logit_gate_max = 1.2
        self.comp_sub_warmup_epochs = max(0, int(comp_sub_warmup_epochs))
        self.comp_sub_ramp_epochs = max(1, int(comp_sub_ramp_epochs))
        self.rel_conf_comp_scale = rel_conf_comp_scale
        self.rel_conf_sub_scale = rel_conf_sub_scale
        self.rel_conf_event_gain = rel_conf_event_gain
        self.rel_conf_repeat_penalty = rel_conf_repeat_penalty
        self.rel_conf_len_gain = rel_conf_len_gain
        self.enable_comp_branch = bool(enable_comp_branch)
        self.enable_sub_branch = bool(enable_sub_branch)
        self.enable_logit_residual = bool(enable_logit_residual)
        self.enable_rel_conf_gate = bool(enable_rel_conf_gate)
        self.rel_conf_sub_short_gain = 0.35
        self.rel_conf_sub_repeat_gain = 0.45
        self.rel_conf_sub_event_gain = 0.20
        self.rel_conf_sub_bias = -0.20
        self.default_session_len = 3.0
        self.default_event_strength = 0.5
        self._position_weight_cache = dict()
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            if weight is not None and weight.data is not None:
                weight.data.uniform_(-stdv, stdv)

        # 恢复行为强度先验（防止被 uniform 初始化覆盖）
        with torch.no_grad():
            d = self.event_scale.weight.device
            self.event_scale.weight.data[0] = torch.tensor([0.0], device=d)
            self.event_scale.weight.data[1] = torch.tensor([0.5], device=d)
            self.event_scale.weight.data[2] = torch.tensor([1.2], device=d)
            self.event_scale.weight.data[3] = torch.tensor([2.0], device=d)
    def trans_adj(self, adjacency):
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))
    def calc_repeat_ratio_batch(self, session_item):
        ratios = []
        for sess in session_item.detach().cpu().numpy():
            seq = [x for x in sess.tolist() if x > 0]
            ratios.append(0.0 if len(seq) == 0 else (len(seq) - len(set(seq))) / len(seq))
        return torch.tensor(ratios, device=session_item.device, dtype=torch.float32)
    # 机制3：关系置信先验
    def build_fuzzy_relation_prior(self, session_item, reversed_sess_event):
        repeat_ratio = self.calc_repeat_ratio_batch(session_item)
        evt_strength = self.event_scale(reversed_sess_event).squeeze(-1).mean(dim=1)
        if self.dataset == 'Tmall':
            c1 = torch.clamp(0.70 + 0.20 * evt_strength - 0.15 * repeat_ratio, min=0.10)  # 顺序
            c2 = torch.clamp(0.60 + 0.15 * (1.0 - repeat_ratio), min=0.10)  # 转移
            c3 = torch.clamp(0.30 + 0.35 * repeat_ratio + 0.10 * evt_strength, min=0.10)  # 共现
        else:
            c1 = torch.clamp(0.65 + 0.15 * evt_strength - 0.20 * repeat_ratio, min=0.05)
            c2 = torch.clamp(0.55 + 0.10 * (1.0 - repeat_ratio), min=0.05)
            c3 = torch.clamp(0.35 + 0.30 * repeat_ratio + 0.05 * evt_strength, min=0.05)
        prior = torch.stack([c1, c2, c3], dim=1)
        prior = prior / (prior.sum(dim=1, keepdim=True) + 1e-8)
        return prior

    def get_dynamic_fuzzy_gate(self, sess_emb):
        gate_logits = self.gate_mlp(sess_emb)
        gate_logits = self.gate_dropout(gate_logits)
        return torch.softmax(gate_logits, dim=-1)

    # 机制2：动态超边激活概率
    def build_hyperedge_activation(self, session_item, reversed_sess_event):
        """Compute session-level hyperedge activation probabilities from repeat ratio and event intensity."""
        repeat_ratio = self.calc_repeat_ratio_batch(session_item)
        evt_strength = self.event_scale(reversed_sess_event).squeeze(-1).mean(dim=1)
        act = self.hyperedge_min_prob + self.hyperedge_event_gain * evt_strength - self.hyperedge_repeat_penalty * repeat_ratio
        return torch.clamp(act, min=0.10, max=0.95)

    def normalize_item_prior(self, prior):
        """Normalize item prior tensor by mean value with numerical-stability epsilon."""
        scale = torch.clamp(prior.abs().mean(), min=self.numerical_eps)
        return prior / scale

    def compute_comp_sub_weights(self, repeat_ratio, session_len=None, event_strength=None):
        if session_len is None:
            session_len = torch.ones_like(repeat_ratio) * self.default_session_len
        if event_strength is None:
            event_strength = torch.ones_like(repeat_ratio) * self.default_event_strength
        short_factor = torch.clamp(1.0 / torch.sqrt(session_len.float().clamp(min=1.0)), min=0.2, max=1.0)
        event_factor = torch.clamp(event_strength.float(), min=0.0, max=2.0)
        comp_adapt = torch.clamp(1.0 + 0.15 * event_factor - 0.10 * short_factor, min=0.85, max=1.25)
        sub_adapt = torch.clamp(1.0 + 0.35 * short_factor + 0.08 * repeat_ratio, min=0.90, max=1.45)
        comp_w = torch.clamp(
            (self.comp_weight_base - self.comp_weight_decay * repeat_ratio) * comp_adapt,
            min=self.comp_weight_min,
            max=self.comp_weight_max
        )
        sub_w = torch.clamp(
            (self.sub_weight_base + self.sub_weight_gain * repeat_ratio) * sub_adapt,
            min=self.sub_weight_min,
            max=self.sub_weight_max
        )
        base_w = torch.clamp(
            1.0 - comp_w - sub_w,
            min=self.base_weight_min,
            max=self.base_weight_max
        )
        w_sum = torch.clamp(base_w + comp_w + sub_w, min=self.numerical_eps)
        return base_w / w_sum, comp_w / w_sum, sub_w / w_sum

    def _build_position_weight(self, seq_len, device):
        device_key = (device.type, device.index if device.type == "cuda" else -1)
        key = (seq_len, device_key)
        if key not in self._position_weight_cache:
            pos_ids = torch.arange(seq_len, device=device).float()
            self._position_weight_cache[key] = torch.exp(-self.pos_decay * pos_ids).view(1, seq_len, 1)
        return self._position_weight_cache[key]

    def fuzzy_cross_view(self, h1, h2, h3):
        channel_embeddings = [h1, h2, h3]
        raw_weights = []
        for emb in channel_embeddings:
            score = torch.sum(torch.mul(self.attention, torch.matmul(emb, self.attention_mat)), 1)
            raw_weights.append(score)
        raw_weights = torch.stack(raw_weights, dim=1)
        score = torch.softmax(raw_weights, dim=-1)
        mixed = score[:, 0:1] * h1 + score[:, 1:2] * h2 + score[:, 2:3] * h3
        return mixed, score

    def fuse_session_views(self, s1, s2, s3, gate):
        gate = gate / (gate.sum(dim=1, keepdim=True) + 1e-8)
        return gate[:, 0:1] * s1 + gate[:, 1:2] * s2 + gate[:, 2:3] * s3

    def ce_with_label_smoothing(self, logits, target, smooth=0.0):
        n_class = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(smooth / (n_class - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smooth)
        log_prob = F.log_softmax(logits, dim=1)
        return torch.mean(torch.sum(-true_dist * log_prob, dim=1))

    def bpr_hard_negative_loss(self, logits, target, topk=100):
        # logits: [B, N], target: [B]
        B, N = logits.size()
        pos = logits.gather(1, target.view(-1, 1)).squeeze(1)  # [B]

        neg_logits = logits.clone()
        neg_logits.scatter_(1, target.view(-1, 1), -1e9)  # mask positive
        k = min(topk, N - 1)
        hard_vals, hard_idx = torch.topk(neg_logits, k=k, dim=1)  # [B, k]
        # 在 hard negatives 中随机采一个，避免总盯最难负样本导致不稳定
        rand_col = torch.randint(0, k, (B,), device=logits.device)
        sampled_neg = hard_vals[torch.arange(B, device=logits.device), rand_col]
        return -torch.log(torch.sigmoid(pos - sampled_neg) + 1e-8).mean()

    def generate_sess_emb(self, item_embedding, event_embedding, session_item, session_len, reversed_sess_item,
                          reversed_sess_event, mask):
        zeros = torch.zeros(1, self.emb_size, device=item_embedding.device)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        event_embedding = torch.cat([zeros, event_embedding], 0)
        batch_size = session_item.shape[0]
        seq_len_all = list(reversed_sess_item.shape)[1]
        item_part = item_embedding[reversed_sess_item]
        event_part = event_embedding[reversed_sess_event]
        event_scales = self.event_scale(reversed_sess_event).to(item_embedding.device)
        position_weight = self._build_position_weight(seq_len_all, item_embedding.device)
        seq_h = position_weight * (item_part + event_scales * event_part)
        if session_len.dim() == 1:
            session_len = session_len.unsqueeze(1)
        hs = torch.sum(seq_h, 1) / (session_len.float() + 1e-8)
        mask = mask.float().unsqueeze(-1)
        len_seq = seq_h.shape[1]
        pos_len = min(len_seq, self.pos_len)
        pos_emb = self.pos_embedding.weight[:pos_len]
        if len_seq > self.pos_len:
            seq_h = seq_h[:, -self.pos_len:, :]
            mask = mask[:, -self.pos_len:, :]
            len_seq = self.pos_len
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        hs = hs.unsqueeze(1).repeat(1, len_seq, 1)

        m = min(seq_h.shape[1], pos_emb.shape[1], mask.shape[1], hs.shape[1])
        seq_h, pos_emb, mask, hs = seq_h[:, :m], pos_emb[:, :m], mask[:, :m], hs[:, :m]
        nh = torch.matmul(torch.cat([pos_emb, seq_h], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2) * mask
        return torch.sum(beta * seq_h, 1)

    def generate_sess_emb_npos(self, item_embedding, event_embedding, session_item, session_len, reversed_sess_item,
                               reversed_sess_event, mask):
        zeros = torch.zeros(1, self.emb_size, device=item_embedding.device)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        event_embedding = torch.cat([zeros, event_embedding], 0)
        batch_size = session_item.shape[0]
        seq_len_all = list(reversed_sess_item.shape)[1]
        item_part = item_embedding[reversed_sess_item]
        event_part = event_embedding[reversed_sess_event]
        event_scales = self.event_scale(reversed_sess_event).to(item_embedding.device)
        position_weight = self._build_position_weight(seq_len_all, item_embedding.device)
        seq_h = position_weight * (item_part + event_scales * event_part)
        if session_len.dim() == 1:
            session_len = session_len.unsqueeze(1)
        hs = torch.sum(seq_h, 1) / (session_len.float() + 1e-8)
        mask = mask.float().unsqueeze(-1)
        len_seq = seq_h.shape[1]
        hs = hs.unsqueeze(1).repeat(1, len_seq, 1)

        m = min(seq_h.shape[1], mask.shape[1], hs.shape[1])
        seq_h, mask, hs = seq_h[:, :m], mask[:, :m], hs[:, :m]
        nh = torch.sigmoid(self.glu1(seq_h) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2) * mask
        return torch.sum(beta * seq_h, 1)

    # 机制4：多粒度模糊损失
    def compute_fuzzy_losses(self, scores_item, tar, s1, s2, s3, sf, gate):
        rel_loss = (
            (gate[:, 0:1] * (s1 - sf).pow(2)).mean() +
            (gate[:, 1:2] * (s2 - sf).pow(2)).mean() +
            (gate[:, 2:3] * (s3 - sf).pow(2)).mean()
        )

        entropy = -torch.sum(gate * torch.log(gate + 1e-8), dim=1) / math.log(3.0)
        ce_each = F.cross_entropy(scores_item, tar, reduction='none')
        sess_loss = torch.mean((1.0 + 0.15 * entropy) * ce_each)

        pos = scores_item.gather(1, tar.view(-1, 1)).squeeze(1)
        neg = scores_item.clone()
        neg.scatter_(1, tar.view(-1, 1), -1e9)
        hard_neg, _ = torch.max(neg, dim=1)
        rank_loss = torch.mean(F.relu(self.rank_margin - pos + hard_neg))

        n_items = scores_item.size(1)
        smooth = torch.clamp(
            self.soft_label_smooth_min + self.soft_label_smooth_gain * entropy,
            min=self.soft_label_smooth_min,
            max=self.soft_label_smooth_max
        ).unsqueeze(1)
        one_hot = F.one_hot(tar, num_classes=n_items).float()
        uniform = torch.full_like(one_hot, 1.0 / n_items)
        soft_targets = (1.0 - smooth) * one_hot + smooth * uniform
        soft_loss = -(soft_targets * F.log_softmax(scores_item, dim=1)).sum(dim=1).mean()

        hyper_loss = ((s1 - s2).pow(2).mean() + (s2 - s3).pow(2).mean() + (s1 - s3).pow(2).mean()) / 3.0

        fuzzy_loss = (
            self.rel_loss_weight * rel_loss +
            self.sess_loss_weight * sess_loss +
            self.rank_loss_weight * rank_loss +
            self.hyper_loss_weight * hyper_loss +
            self.soft_loss_weight * soft_loss
        )
        return fuzzy_loss

    def fuzzy_schedule(self, epoch):
        if epoch < self.warmup_epochs:
            return 0.0
        span = 4.0
        factor = min((epoch - self.warmup_epochs + 1) / span, 1.0)
        return self.max_fuzzy_factor * factor

    def relation_branch_schedule(self, epoch):
        if epoch < self.comp_sub_warmup_epochs:
            return 0.0
        return min((epoch - self.comp_sub_warmup_epochs + 1) / float(self.comp_sub_ramp_epochs), 1.0)

    def relation_reliability(self, repeat_ratio, session_len, event_strength):
        repeat_ratio = torch.clamp(repeat_ratio.float(), min=0.0, max=1.0)
        session_len = torch.clamp(session_len.float(), min=1.0)
        event_strength = torch.clamp(event_strength.float(), min=0.0, max=2.0)
        short_factor = torch.clamp(1.0 / torch.sqrt(session_len), min=0.2, max=1.0)
        long_factor = 1.0 - short_factor
        comp_logit = self.rel_conf_comp_scale * (
            self.rel_conf_event_gain * event_strength + self.rel_conf_len_gain * long_factor - self.rel_conf_repeat_penalty * repeat_ratio
        )
        sub_logit = self.rel_conf_sub_scale * (
            self.rel_conf_sub_short_gain * short_factor +
            self.rel_conf_sub_repeat_gain * repeat_ratio +
            self.rel_conf_sub_event_gain * event_strength +
            self.rel_conf_sub_bias
        )
        comp_rel = torch.clamp(torch.sigmoid(comp_logit), min=0.05, max=1.0)
        sub_rel = torch.clamp(torch.sigmoid(sub_logit), min=0.05, max=1.0)
        return comp_rel, sub_rel

    def forward(self, session_item, session_len, reversed_sess_item, reversed_sess_event, mask, epoch, tar, train):
        repeat_ratio = self.calc_repeat_ratio_batch(session_item)
        relation_progress = self.relation_branch_schedule(epoch)
        fuzzy_strength = (0.20 + 0.45 * torch.clamp(repeat_ratio / 0.3, max=1.0)).unsqueeze(1)
        sess_len_vec = session_len.float().squeeze(-1).clamp(min=1.0)
        event_strength_vec = self.event_scale(reversed_sess_event).squeeze(-1).mean(dim=1)
        comp_rel_conf_sess, sub_rel_conf_sess = self.relation_reliability(repeat_ratio, sess_len_vec, event_strength_vec)
        if not self.enable_rel_conf_gate:
            comp_rel_conf_sess = torch.ones_like(comp_rel_conf_sess)
            sub_rel_conf_sess = torch.ones_like(sub_rel_conf_sess)
        event_weight = self.event_embedding.weight
        i1, _ = self.ItemGraph(self.adj1_fuzzy, self.adjacency_fuzzy, self.embedding1.weight, 0)
        i2, _ = self.ItemGraph(self.adj2_fuzzy, self.adjacency_T_fuzzy, self.embedding2.weight, 1)
        i3_base, _ = self.ItemGraph(self.R1, self.adjacency1, self.embedding3.weight, 2)
        i3_fuzzy, _ = self.ItemGraph(self.R1_fuzzy, self.adjacency1_fuzzy, self.embedding3.weight, 2)
        i_comp_pair, _ = self.ItemGraph(self.comp_deg, self.adjacency_comp, self.embedding3.weight, 2)
        i_sub_pair, _ = self.ItemGraph(self.sub_deg, self.adjacency_sub, self.embedding3.weight, 2)
        i_comp_hyper = self.CompHyperGraph(self.hyper_comp, self.embedding3.weight)
        i_sub_hyper = self.SubHyperGraph(self.hyper_sub, self.embedding3.weight)
        i_comp = (1.0 - self.comp_sub_pair_hyper_mix) * i_comp_pair + self.comp_sub_pair_hyper_mix * i_comp_hyper
        i_sub = (1.0 - self.comp_sub_pair_hyper_mix) * i_sub_pair + self.comp_sub_pair_hyper_mix * i_sub_hyper
        hyperedge_act = self.build_hyperedge_activation(session_item, reversed_sess_event)
        mean_hyperedge_activation = hyperedge_act.mean()
        item_hyper_prior_norm = self.normalize_item_prior(self.R_fuzzy)
        i3 = mean_hyperedge_activation * i3_fuzzy + (1.0 - mean_hyperedge_activation) * i3_base
        # item-level relation blend uses batch-aggregated session-aware gate
        base_weight, comp_weight, sub_weight = self.compute_comp_sub_weights(
            repeat_ratio.mean(), sess_len_vec.mean(), event_strength_vec.mean()
        )
        if not self.enable_comp_branch:
            comp_weight = torch.zeros_like(comp_weight)
        if not self.enable_sub_branch:
            sub_weight = torch.zeros_like(sub_weight)
        comp_weight = comp_weight * comp_rel_conf_sess.mean() * relation_progress
        sub_weight = sub_weight * sub_rel_conf_sess.mean() * relation_progress
        base_weight = torch.clamp(1.0 - comp_weight - sub_weight, min=self.base_weight_min, max=self.base_weight_max)
        weight_sum = torch.clamp(base_weight + comp_weight + sub_weight, min=self.numerical_eps)
        base_weight, comp_weight, sub_weight = base_weight / weight_sum, comp_weight / weight_sum, sub_weight / weight_sum
        i3 = base_weight * i3 + comp_weight * i_comp + sub_weight * i_sub
        i3 = i3 * ((1.0 - self.item_prior_mix) + self.item_prior_mix * item_hyper_prior_norm.unsqueeze(1))
        i1, i2, i3 = F.normalize(i1, dim=-1), F.normalize(i2, dim=-1), F.normalize(i3, dim=-1)
        item_mix, _ = self.fuzzy_cross_view(i1, i2, i3)
        # Penalize correlation between comp/sub channels (encourage orthogonality).
        comp_sub_orthogonality_loss = (F.normalize(i_comp, dim=-1) * F.normalize(i_sub, dim=-1)).sum(dim=1).pow(2).mean()

        if self.dataset == 'Tmall':
            s1 = self.generate_sess_emb_npos(i1, event_weight, session_item, session_len, reversed_sess_item, reversed_sess_event, mask)
            s2 = self.generate_sess_emb_npos(i2, event_weight, session_item, session_len, reversed_sess_item, reversed_sess_event, mask)
            s3_base = self.generate_sess_emb_npos(i3_base, event_weight, session_item, session_len, reversed_sess_item,reversed_sess_event, mask)
            s3_fuzzy = self.generate_sess_emb_npos(i3_fuzzy, event_weight, session_item, session_len,reversed_sess_item, reversed_sess_event, mask)
            s_comp = self.generate_sess_emb_npos(i_comp, event_weight, session_item, session_len, reversed_sess_item,reversed_sess_event, mask)
            s_sub = self.generate_sess_emb_npos(i_sub, event_weight, session_item, session_len, reversed_sess_item,reversed_sess_event, mask)
        else:
            s1 = self.generate_sess_emb(i1, event_weight, session_item, session_len, reversed_sess_item, reversed_sess_event, mask)
            s2 = self.generate_sess_emb(i2, event_weight, session_item, session_len, reversed_sess_item, reversed_sess_event, mask)
            s3_base = self.generate_sess_emb(i3_base, event_weight, session_item, session_len, reversed_sess_item,reversed_sess_event, mask)
            s3_fuzzy = self.generate_sess_emb(i3_fuzzy, event_weight, session_item, session_len, reversed_sess_item,reversed_sess_event, mask)
            s_comp = self.generate_sess_emb(i_comp, event_weight, session_item, session_len, reversed_sess_item,reversed_sess_event, mask)
            s_sub = self.generate_sess_emb(i_sub, event_weight, session_item, session_len, reversed_sess_item,reversed_sess_event, mask)

        s3 = hyperedge_act.unsqueeze(1) * s3_fuzzy + (1.0 - hyperedge_act.unsqueeze(1)) * s3_base
        # session-level uses per-session repeat ratio for personalized relation fusion
        base_w_sess, comp_w_sess, sub_w_sess = self.compute_comp_sub_weights(repeat_ratio, sess_len_vec, event_strength_vec)
        if not self.enable_comp_branch:
            comp_w_sess = torch.zeros_like(comp_w_sess)
        if not self.enable_sub_branch:
            sub_w_sess = torch.zeros_like(sub_w_sess)
        comp_w_sess = comp_w_sess * comp_rel_conf_sess * relation_progress
        sub_w_sess = sub_w_sess * sub_rel_conf_sess * relation_progress
        base_w_sess = torch.clamp(1.0 - comp_w_sess - sub_w_sess, min=self.base_weight_min, max=self.base_weight_max)
        sess_w_sum = torch.clamp(base_w_sess + comp_w_sess + sub_w_sess, min=self.numerical_eps)
        base_w_sess, comp_w_sess, sub_w_sess = (
            base_w_sess / sess_w_sum,
            comp_w_sess / sess_w_sum,
            sub_w_sess / sess_w_sum
        )
        base_w, comp_w, sub_w = base_w_sess.unsqueeze(1), comp_w_sess.unsqueeze(1), sub_w_sess.unsqueeze(1)
        s3 = base_w * s3 + comp_w * s_comp + sub_w * s_sub

        prior_gate = self.build_fuzzy_relation_prior(session_item, reversed_sess_event)
        learned_gate = self.get_dynamic_fuzzy_gate((s1 + s2 + s3) / 3.0)
        gate = (1.0 - self.fuzzy_prior_strength) * learned_gate + self.fuzzy_prior_strength * prior_gate

        uniform = torch.ones_like(gate) / 3.0
        gate = fuzzy_strength * gate + (1.0 - fuzzy_strength) * uniform
        gate = 0.98 * gate + 0.02 * uniform
        gate = gate / (gate.sum(dim=1, keepdim=True) + 1e-8)

        sf = self.fuse_session_views(s1, s2, s3, gate)
        sf_base = sf

        last_item_ids = reversed_sess_item[:, 0]
        valid_last = (last_item_ids > 0)
        last_item_emb = torch.zeros(session_item.size(0), self.emb_size, device=item_mix.device)
        if valid_last.any():
            last_item_pos = torch.clamp(last_item_ids[valid_last] - 1, min=0, max=self.n_node - 1)
            last_item_emb[valid_last] = item_mix[last_item_pos]
        len_factor = torch.clamp(
            1.0 / torch.sqrt(session_len.float().squeeze(-1).clamp(min=1.0)),
            min=self.short_len_factor_min, max=1.0
        ).unsqueeze(1)
        short_gate = torch.sigmoid(self.short_intent_mlp(torch.cat([sf, last_item_emb], dim=1)))
        short_gate = self.short_intent_min + (self.short_intent_max - self.short_intent_min) * short_gate * len_factor
        short_gate = short_gate * valid_last.float().unsqueeze(1)
        sf = (1.0 - short_gate) * sf + short_gate * last_item_emb

        if train:
            sf = self.final_dropout(sf)

        sf_norm = F.normalize(sf, dim=-1)
        sf = self.w_k * sf_norm
        item_mix = F.normalize(item_mix, dim=-1)
        i_comp_norm = F.normalize(i_comp, dim=-1)
        i_sub_norm = F.normalize(i_sub, dim=-1)
        scores_base = torch.mm(sf, item_mix.t())
        scores_comp = torch.mm(sf, i_comp_norm.t())
        scores_sub = torch.mm(sf, i_sub_norm.t())
        comp_logit_gate = self.logit_comp_scale * comp_w_sess.unsqueeze(1)
        sub_logit_gate = self.logit_sub_scale * (
            sub_w_sess.unsqueeze(1) + self.logit_short_sub_boost * short_gate
        )
        sub_logit_gate = torch.clamp(sub_logit_gate, min=0.0, max=self.sub_logit_gate_max)
        if self.enable_logit_residual:
            scores_item = scores_base + comp_logit_gate * scores_comp + sub_logit_gate * scores_sub
        else:
            scores_item = scores_base

        # temperature scaling for better ranking sharpness
        scores_item = scores_item / self.score_temperature

        tar_safe = torch.clamp(tar, min=0, max=item_mix.size(0) - 1)
        ce_loss = self.ce_with_label_smoothing(scores_item, tar_safe, smooth=self.label_smoothing)
        bpr_loss = self.bpr_hard_negative_loss(scores_item, tar_safe, topk=self.topk_hardneg)
        if train:
            sf_base_norm = F.normalize(sf_base, dim=-1)
            target_item_emb = item_mix[tar_safe]
            intent_align_loss = (1.0 - F.cosine_similarity(sf_base_norm, target_item_emb, dim=-1)).mean()
        else:
            intent_align_loss = torch.tensor(0.0, device=scores_item.device)
        loss_item = ce_loss + self.bpr_loss_weight * bpr_loss + self.intent_align_weight * intent_align_loss
        if train and self.enable_comp_branch and self.enable_sub_branch:
            loss_item = loss_item + (self.comp_sub_decouple_weight * relation_progress) * comp_sub_orthogonality_loss
        con_loss = torch.tensor(0.0, device=scores_item.device)

        if train:
            fuzzy_raw = self.compute_fuzzy_losses(scores_item, tar, s1, s2, s3, sf, gate)
            fuzzy_loss = self.fuzzy_schedule(epoch) * fuzzy_raw
        else:
            fuzzy_loss = torch.tensor(0.0, device=scores_item.device)

        return con_loss, loss_item, scores_item, fuzzy_loss


def forward(model, i, data, epoch, train):
    ret = data.get_slice(i)

    if len(ret) == 5:
        tar, session_len, session_item, reversed_sess_item, mask = ret
        reversed_sess_event = np.zeros_like(reversed_sess_item)
    else:
        tar, session_len, session_item, session_events, reversed_sess_item, reversed_sess_event, mask = ret

    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    reversed_sess_event = trans_to_cuda(torch.Tensor(reversed_sess_event).long())

    con_loss, loss_item, scores_item, fuzzy_loss = model(
        session_item, session_len, reversed_sess_item, reversed_sess_event, mask, epoch, tar, train
    )
    return tar, scores_item, con_loss, loss_item, fuzzy_loss


def train_test(model, train_data, test_data, epoch):
    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    amp_enabled = model.use_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    model.train()
    for i in tqdm(slices):
        model.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            tar, scores_item, con_loss, loss_item, fuzzy_loss = forward(model, i, train_data, epoch, train=True)
            loss = loss_item + con_loss + fuzzy_loss
        scaler.scale(loss).backward()
        scaler.unscale_(model.optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        scaler.step(model.optimizer)
        scaler.update()
        total_loss += loss.item()

    print('\tLoss:\t%.3f' % total_loss)

    top_K = [5, 10, 20, 50]
    max_topk = max(top_K)
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
        metrics['ndcg%d' % K] = []

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    slices = test_data.generate_batch(model.batch_size)

    with torch.no_grad():
        for i in tqdm(slices):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                tar, scores_item, _, _, _ = forward(model, i, test_data, epoch, train=False)
            # Robust for any dataset where catalog size may be smaller than requested max_topk.
            effective_topk = min(max_topk, scores_item.size(1))
            _, index = torch.topk(scores_item, k=effective_topk, dim=1)
            index = trans_to_cpu(index).detach().numpy()
            tar = trans_to_cpu(tar).detach().numpy()

            for K in top_K:
                for prediction, target in zip(index[:, :K], tar):
                    prediction_list = prediction.tolist()
                    DCG = 0.0
                    for j in range(K):
                        if prediction_list[j] == target:
                            DCG += 1 / math.log2(j + 2)
                    metrics['ndcg%d' % K].append(DCG)  # IDCG=1
                    metrics['hit%d' % K].append(np.isin(target, prediction))
                    pos = np.where(prediction == target)[0]
                    metrics['mrr%d' % K].append(0 if len(pos) == 0 else 1 / (pos[0] + 1))

    return metrics, total_loss
