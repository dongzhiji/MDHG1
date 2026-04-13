import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from numba import jit
from tqdm import tqdm


def trans_to_cuda(variable):
    return variable.cuda() if torch.cuda.is_available() else variable


def trans_to_cpu(variable):
    return variable.cpu() if torch.cuda.is_available() else variable


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
            item_embeddings = torch.sparse.mm(adjacency, item_embeddings)

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


class MDHG(Module):
    def __init__(self, R, adj1, adj2, adjacency, adjacency_T, adjacency1, R1,
                 adjacency_fuzzy, adjacency_T_fuzzy, adjacency1_fuzzy,
                 adj1_fuzzy, adj2_fuzzy, R_fuzzy, R1_fuzzy,
                 n_node, lr, layers, l2, beta, lam, eps, dataset,
                 K1, K2, K3, dropout, alpha, emb_size=100, batch_size=100):
        super(MDHG, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.dataset = dataset
        self.lr = lr
        self.layers = layers
        self.w_k = 10

        self.adjacency = trans_to_cuda(self.trans_adj(adjacency))
        self.adjacency_T = trans_to_cuda(self.trans_adj(adjacency_T))
        self.adjacency1 = trans_to_cuda(self.trans_adj(adjacency1))
        self.adjacency_fuzzy = trans_to_cuda(self.trans_adj(adjacency_fuzzy))
        self.adjacency_T_fuzzy = trans_to_cuda(self.trans_adj(adjacency_T_fuzzy))
        self.adjacency1_fuzzy = trans_to_cuda(self.trans_adj(adjacency1_fuzzy))

        self.adj1 = torch.cuda.FloatTensor(adj1) if torch.cuda.is_available() else torch.FloatTensor(adj1)
        self.adj2 = torch.cuda.FloatTensor(adj2) if torch.cuda.is_available() else torch.FloatTensor(adj2)
        self.R1 = torch.cuda.FloatTensor(R1) if torch.cuda.is_available() else torch.FloatTensor(R1)
        self.adj1_fuzzy = torch.cuda.FloatTensor(adj1_fuzzy) if torch.cuda.is_available() else torch.FloatTensor(adj1_fuzzy)
        self.adj2_fuzzy = torch.cuda.FloatTensor(adj2_fuzzy) if torch.cuda.is_available() else torch.FloatTensor(adj2_fuzzy)
        self.R_fuzzy = torch.cuda.FloatTensor(R_fuzzy) if torch.cuda.is_available() else torch.FloatTensor(R_fuzzy)
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

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # ---- 稳定版损失配置（先保主任务）----
        self.fuzzy_prior_strength = 0.35
        self.rank_margin = 0.30
        self.rel_loss_weight = 0.05
        self.sess_loss_weight = 0.01
        self.rank_loss_weight = 0.15
        self.hyper_loss_weight = 0.01
        self.soft_loss_weight = 0.05

        self.warmup_epochs = 2
        self.max_fuzzy_factor = 0.25
        self.hyperedge_min_prob = 0.35
        self.hyperedge_event_gain = 0.20
        self.hyperedge_repeat_penalty = 0.30

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

        c1 = torch.clamp(0.65 + 0.15 * evt_strength - 0.20 * repeat_ratio, min=0.05)  # 顺序
        c2 = torch.clamp(0.55 + 0.10 * (1.0 - repeat_ratio), min=0.05)               # 转移
        c3 = torch.clamp(0.35 + 0.30 * repeat_ratio + 0.05 * evt_strength, min=0.05) # 共现

        prior = torch.stack([c1, c2, c3], dim=1)
        prior = prior / (prior.sum(dim=1, keepdim=True) + 1e-8)
        return prior

    def get_dynamic_fuzzy_gate(self, sess_emb):
        gate_logits = self.gate_mlp(sess_emb)
        gate_logits = self.gate_dropout(gate_logits)
        return torch.softmax(gate_logits, dim=-1)

    # 机制2：动态超边激活概率
    def build_hyperedge_activation(self, session_item, reversed_sess_event):
        repeat_ratio = self.calc_repeat_ratio_batch(session_item)
        evt_strength = self.event_scale(reversed_sess_event).squeeze(-1).mean(dim=1)
        act = self.hyperedge_min_prob + self.hyperedge_event_gain * evt_strength - self.hyperedge_repeat_penalty * repeat_ratio
        return torch.clamp(act, min=0.10, max=0.95)

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

    def generate_sess_emb(self, item_embedding, event_embedding, session_item, session_len, reversed_sess_item,
                          reversed_sess_event, mask):
        zeros = torch.zeros(1, self.emb_size, device=item_embedding.device)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        event_embedding = torch.cat([zeros, event_embedding], 0)

        batch_size = session_item.shape[0]
        seq_len_all = list(reversed_sess_item.shape)[1]
        seq_h = torch.zeros(batch_size, seq_len_all, self.emb_size, device=item_embedding.device)

        for i in range(batch_size):
            item_part = item_embedding[reversed_sess_item[i]]
            event_part = event_embedding[reversed_sess_event[i]]
            event_scales = self.event_scale(reversed_sess_event[i]).to(item_embedding.device)
            seq_len = item_part.shape[0]
            pos_ids = torch.arange(seq_len, device=item_part.device).float()
            position_weight = torch.exp(-self.pos_decay * pos_ids).unsqueeze(-1)
            seq_h[i] = position_weight * (item_part + event_scales * event_part)

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
        seq_h = torch.zeros(batch_size, seq_len_all, self.emb_size, device=item_embedding.device)

        for i in range(batch_size):
            item_part = item_embedding[reversed_sess_item[i]]
            event_part = event_embedding[reversed_sess_event[i]]
            event_scales = self.event_scale(reversed_sess_event[i]).to(item_embedding.device)

            seq_len = item_part.shape[0]
            pos_ids = torch.arange(seq_len, device=item_part.device).float()
            position_weight = torch.exp(-self.pos_decay * pos_ids).unsqueeze(-1)
            seq_h[i] = position_weight * (item_part + event_scales * event_part)

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
        smooth = torch.clamp(0.02 + 0.08 * entropy, min=0.02, max=0.10).unsqueeze(1)
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

    def forward(self, session_item, session_len, reversed_sess_item, reversed_sess_event, mask, epoch, tar, train):
        repeat_ratio = self.calc_repeat_ratio_batch(session_item)
        fuzzy_strength = (0.20 + 0.45 * torch.clamp(repeat_ratio / 0.3, max=1.0)).unsqueeze(1)

        event_weight = self.event_embedding.weight

        i1, _ = self.ItemGraph(self.adj1_fuzzy, self.adjacency_fuzzy, self.embedding1.weight, 0)
        i2, _ = self.ItemGraph(self.adj2_fuzzy, self.adjacency_T_fuzzy, self.embedding2.weight, 1)
        i3_base, _ = self.ItemGraph(self.R1, self.adjacency1, self.embedding3.weight, 2)
        i3_fuzzy, _ = self.ItemGraph(self.R1_fuzzy, self.adjacency1_fuzzy, self.embedding3.weight, 2)
        hyperedge_act = self.build_hyperedge_activation(session_item, reversed_sess_event).mean().view(1, 1)
        i3 = hyperedge_act * i3_fuzzy + (1.0 - hyperedge_act) * i3_base
        item_hyper_prior = self.R_fuzzy.squeeze(0)
        item_hyper_prior = item_hyper_prior / (item_hyper_prior.mean() + 1e-8)
        i3 = i3 * (0.9 + 0.1 * item_hyper_prior.unsqueeze(1))
        i1, i2, i3 = F.normalize(i1, dim=-1), F.normalize(i2, dim=-1), F.normalize(i3, dim=-1)

        item_mix, _ = self.fuzzy_cross_view(i1, i2, i3)

        if self.dataset == 'Tmall':
            s1 = self.generate_sess_emb_npos(i1, event_weight, session_item, session_len, reversed_sess_item, reversed_sess_event, mask)
            s2 = self.generate_sess_emb_npos(i2, event_weight, session_item, session_len, reversed_sess_item, reversed_sess_event, mask)
            s3 = self.generate_sess_emb_npos(i3, event_weight, session_item, session_len, reversed_sess_item, reversed_sess_event, mask)
        else:
            s1 = self.generate_sess_emb(i1, event_weight, session_item, session_len, reversed_sess_item, reversed_sess_event, mask)
            s2 = self.generate_sess_emb(i2, event_weight, session_item, session_len, reversed_sess_item, reversed_sess_event, mask)
            s3 = self.generate_sess_emb(i3, event_weight, session_item, session_len, reversed_sess_item, reversed_sess_event, mask)

        prior_gate = self.build_fuzzy_relation_prior(session_item, reversed_sess_event)
        learned_gate = self.get_dynamic_fuzzy_gate((s1 + s2 + s3) / 3.0)
        gate = (1.0 - self.fuzzy_prior_strength) * learned_gate + self.fuzzy_prior_strength * prior_gate

        uniform = torch.ones_like(gate) / 3.0
        gate = fuzzy_strength * gate + (1.0 - fuzzy_strength) * uniform
        gate = 0.98 * gate + 0.02 * uniform
        gate = gate / (gate.sum(dim=1, keepdim=True) + 1e-8)

        sf = self.fuse_session_views(s1, s2, s3, gate)
        if train:
            sf = self.final_dropout(sf)

        sf = self.w_k * F.normalize(sf, dim=-1)
        item_mix = F.normalize(item_mix, dim=-1)
        scores_item = torch.mm(sf, item_mix.t())

        loss_item = self.loss_function(scores_item, tar)
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


@jit(nopython=True)
def find_k_largest(K, candidates):
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((iid, score))
    n_candidates.sort(key=lambda d: d[1], reverse=True)
    k_largest_scores = [item[1] for item in n_candidates]
    ids = [item[0] for item in n_candidates]
    for iid, score in enumerate(candidates):
        ind = K
        l = 0
        r = K - 1
        if k_largest_scores[r] < score:
            while r >= l:
                mid = int((r - l) / 2) + l
                if k_largest_scores[mid] >= score:
                    l = mid + 1
                else:
                    r = mid - 1
                if r < l:
                    ind = r
                    break
        if ind < K - 2:
            k_largest_scores[ind + 2:] = k_largest_scores[ind + 1:-1]
            ids[ind + 2:] = ids[ind + 1:-1]
        if ind < K - 1:
            k_largest_scores[ind + 1] = score
            ids[ind + 1] = iid
    return ids


def train_test(model, train_data, test_data, epoch):
    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)

    model.train()
    for i in tqdm(slices):
        model.zero_grad()
        tar, scores_item, con_loss, loss_item, fuzzy_loss = forward(model, i, train_data, epoch, train=True)
        loss = loss_item + con_loss + fuzzy_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        model.optimizer.step()
        total_loss += loss.item()

    print('\tLoss:\t%.3f' % total_loss)

    top_K = [5, 10, 20, 50]
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
            tar, scores_item, _, _, _ = forward(model, i, test_data, epoch, train=False)
            scores = trans_to_cpu(scores_item).detach().numpy()
            index = []
            for idd in range(scores.shape[0]):
                index.append(find_k_largest(50, scores[idd]))
            index = np.array(index)
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
