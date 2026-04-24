import numpy as np
from scipy.sparse import coo_matrix
import warnings
EPSILON = 1e-8
COMP_SUB_DEFAULTS = {
    'max_gap': 3,
    'comp_topk': 8,
    'sub_topk': 8,
    'min_support': 2,
    'sub_context_min': 2,
    'conflict_margin': 0.05
}

def _ordered_pair(i, j):
    return (i, j) if i < j else (j, i)

def fuzzy_membership(x, center=1.0, scale=1.0):
    return float(np.exp(-abs(x - center) / max(scale, 1e-8)))

# ===== 新增：关系专属模糊边权辅助函数 =====
def _event_strength(e):
    # 0 padding, 1 view, 2 addtocart, 3 transaction
    mapping = {0: 0.0, 1: 0.5, 2: 1.0, 3: 2.0}
    return mapping.get(int(e), 0.5)

def _safe_get_event(events, idx):
    if events is None:
        return 1
    if idx < 0 or idx >= len(events):
        return 1
    return events[idx]

def _relation_fuzzy_weight(freq, pos_i, sess_len, e_i, e_j, relation='r1', repeat_ratio=0.0, dataset='Tmall'):
    # Tmall数据集特定参数
    if dataset == 'Tmall':
        # 1) 频次隶属度 - Tmall中高频共现更重要
        if relation == 'r3':
            mu_freq = fuzzy_membership(freq, center=3.0, scale=10.0)
        else:
            mu_freq = fuzzy_membership(freq, center=2.0, scale=6.0)

        # 2) 位置/时序隶属度 - Tmall中近期行为权重更高
        pos_center = max((sess_len - 1) * 0.7, 0.0)
        mu_time = fuzzy_membership(pos_i, center=pos_center, scale=max(sess_len / 4.0, 1.0))

        # 3) 行为强度隶属度 - Tmall中购买行为权重更高
        beh = (_event_strength(e_i) + _event_strength(e_j)) / 2.0
        if relation == 'r1':
            mu_beh = fuzzy_membership(beh, center=1.2, scale=1.0)
        elif relation == 'r2':
            mu_beh = fuzzy_membership(beh, center=1.0, scale=1.2)
        else:
            mu_beh = fuzzy_membership(beh, center=1.5, scale=1.0)

        # 4) 一致性 - Tmall中重复率惩罚降低
        if relation in ('r1', 'r2'):
            mu_cons = 1.0 - min(max(repeat_ratio, 0.0), 1.0) * 0.35
        else:
            mu_cons = 0.65 + min(max(repeat_ratio, 0.0), 1.0) * 0.35
    else:
        # 原始配置
        if relation == 'r3':
            mu_freq = fuzzy_membership(freq, center=2.0, scale=8.0)
        else:
            mu_freq = fuzzy_membership(freq, center=1.0, scale=5.0)

        pos_center = max((sess_len - 1) / 2.0, 0.0)
        mu_time = fuzzy_membership(pos_i, center=pos_center, scale=max(sess_len / 3.0, 1.0))

        beh = (_event_strength(e_i) + _event_strength(e_j)) / 2.0
        if relation == 'r1':
            mu_beh = fuzzy_membership(beh, center=1.0, scale=1.0)
        elif relation == 'r2':
            mu_beh = fuzzy_membership(beh, center=0.8, scale=1.2)
        else:
            mu_beh = fuzzy_membership(beh, center=1.2, scale=1.0)

        if relation in ('r1', 'r2'):
            mu_cons = 1.0 - min(max(repeat_ratio, 0.0), 1.0) * 0.5
        else:
            mu_cons = 0.7 + min(max(repeat_ratio, 0.0), 1.0) * 0.3
    return float(np.clip(mu_freq * mu_time * mu_beh * mu_cons, 1e-8, 1.0))

def data_masks(all_sessions, n_node):
    adj = dict()
    for sess in all_sessions:
        for i, item in enumerate(sess):
            if i == len(sess) - 1:
                break
            current_idx = sess[i] - 1
            next_idx = sess[i + 1] - 1
            if current_idx < 0 or current_idx >= n_node or next_idx < 0 or next_idx >= n_node:
                continue
            if current_idx not in adj:
                adj[current_idx] = dict()
                adj[current_idx][current_idx] = 1
                adj[current_idx][next_idx] = 1
            else:
                if next_idx not in adj[current_idx]:
                    adj[current_idx][next_idx] = 1
                else:
                    adj[current_idx][next_idx] += 1
    for i in range(n_node):
        if i not in adj:
            adj[i] = dict()
            adj[i][i] = 1

    row, col, data = [], [], []
    for i in adj:
        for j in adj[i]:
            if i < n_node and j < n_node:
                row.append(i)
                col.append(j)
                data.append(adj[i][j])
    if len(row) == 0:
        row = list(range(n_node))
        col = list(range(n_node))
        data = [1] * n_node
    return coo_matrix((data, (row, col)), shape=(n_node, n_node))

# ===== 改造：动态超图/关系图模糊边权 =====
def data_masks_fuzzy(all_sessions, n_node, all_events=None):
    adj = dict()
    for s_idx, sess in enumerate(all_sessions):
        events = all_events[s_idx] if all_events is not None and s_idx < len(all_events) else None
        seq = [x for x in sess if x != 0]
        sess_len = max(len(seq), 1)
        repeat_ratio = (len(seq) - len(set(seq))) / max(len(seq), 1)
        for i in range(len(sess) - 1):
            if sess[i] == 0 or sess[i + 1] == 0:
                continue
            current_idx = sess[i] - 1
            next_idx = sess[i + 1] - 1
            if current_idx < 0 or current_idx >= n_node or next_idx < 0 or next_idx >= n_node:
                continue
            ei = _safe_get_event(events, i)
            ej = _safe_get_event(events, i + 1)
            if current_idx not in adj:
                adj[current_idx] = dict()
                adj[current_idx][current_idx] = 1.0
            old = adj[current_idx].get(next_idx, 0.0)
            freq = old + 1.0
            w = _relation_fuzzy_weight(freq, i, sess_len, ei, ej, relation='r1', repeat_ratio=repeat_ratio)
            adj[current_idx][next_idx] = old + w
    for i in range(n_node):
        if i not in adj:
            adj[i] = dict()
        if i not in adj[i]:
            adj[i][i] = 1.0
    row, col, data = [], [], []
    for i in adj:
        for j in adj[i]:
            if i < n_node and j < n_node:
                row.append(i)
                col.append(j)
                data.append(adj[i][j])
    if len(row) == 0:
        row = list(range(n_node))
        col = list(range(n_node))
        data = [1.0] * n_node
    return coo_matrix((data, (row, col)), shape=(n_node, n_node))

def data_R(all_sessions, n_node):
    row, col, data = [], [], []
    edge_idx_s = 0
    for sess in all_sessions:
        for i, item in enumerate(sess):
            row.append(edge_idx_s)
            col.append(item - 1)
            data.append(1)
        edge_idx_s += 1
    return coo_matrix((data, (row, col)), shape=(edge_idx_s, n_node))

# ===== 改造：模糊超边生成（会话级超边） =====
def data_R_fuzzy(all_sessions, n_node, all_events=None):
    row, col, data = [], [], []
    edge_idx_s = 0
    for s_idx, sess in enumerate(all_sessions):
        events = all_events[s_idx] if all_events is not None and s_idx < len(all_events) else None
        seq = [x for x in sess if x != 0]
        sess_len = max(len(seq), 1)
        repeat_ratio = (len(seq) - len(set(seq))) / max(len(seq), 1)
        pos_center = (sess_len - 1) / 2.0
        session_quality = 1.0 - 0.3 * repeat_ratio
        for i, item in enumerate(sess):
            if item == 0:
                continue
            col_idx = item - 1
            if col_idx < 0 or col_idx >= n_node:
                continue
            e_i = _safe_get_event(events, i)
            mu_pos = fuzzy_membership(i, center=pos_center, scale=max(sess_len / 3.0, 1.0))
            mu_evt = fuzzy_membership(_event_strength(e_i), center=1.0, scale=1.0)
            mu_nov = 1.0 - 0.5 * repeat_ratio
            mu_hyper = float(np.clip(mu_pos * mu_evt * mu_nov * session_quality, 1e-8, 1.0))
            row.append(edge_idx_s)
            col.append(col_idx)
            data.append(mu_hyper)
        edge_idx_s += 1
    return coo_matrix((data, (row, col)), shape=(edge_idx_s, n_node))

def data_R1(all_sessions, n_node):
    adj = dict()
    for sess in all_sessions:
        sess_items = [x - 1 for x in sess if x != 0]
        for i in sess_items:
            if i not in adj:
                adj[i] = dict()
            for j in sess_items:
                if j not in adj[i]:
                    adj[i][j] = 1
    row, col, data = [], [], []
    for i in adj:
        for j in adj[i]:
            row.append(i)
            col.append(j)
            data.append(adj[i][j])
    return coo_matrix((data, (row, col)), shape=(n_node, n_node))

def data_R1_fuzzy(all_sessions, n_node, all_events=None):
    adj = dict()
    for s_idx, sess in enumerate(all_sessions):
        events = all_events[s_idx] if all_events is not None and s_idx < len(all_events) else None
        sess_items = [x - 1 for x in sess if x != 0]
        seq = [x for x in sess if x != 0]
        sess_len = max(len(seq), 1)
        repeat_ratio = (len(seq) - len(set(seq))) / max(len(seq), 1)
        for i_pos, i in enumerate(sess_items):
            if i not in adj:
                adj[i] = dict()
            for j_pos, j in enumerate(sess_items):
                old = adj[i].get(j, 0.0)
                freq = old + 1.0
                ei = _safe_get_event(events, i_pos)
                ej = _safe_get_event(events, j_pos)
                w = _relation_fuzzy_weight(freq, i_pos, sess_len, ei, ej, relation='r3', repeat_ratio=repeat_ratio)
                adj[i][j] = old + w
    row, col, data = [], [], []
    for i in adj:
        for j in adj[i]:
            row.append(i)
            col.append(j)
            data.append(adj[i][j])
    return coo_matrix((data, (row, col)), shape=(n_node, n_node))

def _dict_to_coo_with_self_loop(adj, n_node, self_loop=1.0):
    for i in range(n_node):
        if i not in adj:
            adj[i] = dict()
        if i not in adj[i]:
            adj[i][i] = self_loop

    row, col, data = [], [], []
    for i in adj:
        for j, v in adj[i].items():
            if 0 <= i < n_node and 0 <= j < n_node:
                row.append(i)
                col.append(j)
                data.append(v)

    if len(row) == 0:
        row = list(range(n_node))
        col = list(range(n_node))
        data = [self_loop] * n_node
    return coo_matrix((data, (row, col)), shape=(n_node, n_node))


def _build_anchor_item_hyperedges(rel_adj, n_node, topk=8, min_neighbors=1):
    """
    Build item-level hyperedges from anchor-centric relation dict.

    Args:
        rel_adj: dict[int, dict[int, float]], anchor item -> neighbor score map.
        n_node: number of items.
        topk: number of strongest neighbors retained for each anchor.
        min_neighbors: minimum number of retained neighbors to create one hyperedge.

    Returns:
        scipy.sparse.coo_matrix H with shape [n_hyperedges, n_node].
        Each hyperedge contains one anchor item and its top-k related items.
    """
    row, col, data = [], [], []
    edge_idx = 0
    for anchor in range(n_node):
        neighbors = rel_adj.get(anchor, {})
        if len(neighbors) == 0:
            continue

        sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
        sorted_neighbors = [x for x in sorted_neighbors if x[0] != anchor and x[1] > 0][:topk]
        if len(sorted_neighbors) < min_neighbors:
            continue

        max_w = max([w for _, w in sorted_neighbors]) if len(sorted_neighbors) > 0 else 1.0
        row.append(edge_idx)
        col.append(anchor)
        data.append(1.0)

        for nb, w in sorted_neighbors:
            row.append(edge_idx)
            col.append(nb)
            data.append(float(np.clip(w / (max_w + EPSILON), 1e-4, 1.0)))
        edge_idx += 1

    return coo_matrix((data, (row, col)), shape=(edge_idx, n_node))


def _incidence_to_hypergraph_propagation(H, n_node):
    """
    Convert incidence matrix H into node propagation matrix G.

    Formula:
        G = Dv^{-1} * H^T * De^{-1} * H
    where De and Dv are hyperedge/node degree diagonal matrices.

    Args:
        H: incidence matrix with shape [n_hyperedges, n_node].
        n_node: number of items.

    Returns:
        scipy.sparse.coo_matrix G with shape [n_node, n_node].
    """
    if H.shape[0] == 0 or H.nnz == 0:
        idx = np.arange(n_node)
        return coo_matrix((np.ones(n_node), (idx, idx)), shape=(n_node, n_node))

    H = H.tocoo()
    m = H.shape[0]

    edge_deg = np.asarray(H.sum(axis=1)).reshape(-1) + EPSILON
    node_deg = np.asarray(H.sum(axis=0)).reshape(-1) + EPSILON

    edge_idx = np.arange(m)
    node_idx = np.arange(n_node)
    D_e_inv = coo_matrix((1.0 / edge_deg, (edge_idx, edge_idx)), shape=(m, m))
    D_v_inv = coo_matrix((1.0 / node_deg, (node_idx, node_idx)), shape=(n_node, n_node))

    # G = Dv^{-1} * H^T * De^{-1} * H
    G = D_v_inv.dot(H.T.dot(D_e_inv.dot(H))).tocoo()

    # residual self-loop for isolated/rare nodes
    I = coo_matrix((np.ones(n_node), (node_idx, node_idx)), shape=(n_node, n_node))
    G = (0.9 * G + 0.1 * I).tocoo()

    row_sum = np.asarray(G.sum(axis=1)).reshape(-1) + EPSILON
    D_row = coo_matrix((1.0 / row_sum, (node_idx, node_idx)), shape=(n_node, n_node))
    G = D_row.dot(G).tocoo()
    return G


def data_item_hypergraph_comp_sub(all_sessions, n_node, max_gap=3, comp_topk=8, sub_topk=8,
                                  min_support=2, sub_context_min=2, conflict_margin=0.05):
    """
    Build item-view hypergraph relations learned from sessions.

    Args:
        all_sessions: list of session item sequences.
        n_node: total number of items.
        max_gap: maximum distance window for complementary relation extraction.
        comp_topk: top-k neighbors retained per anchor item in complementary hyperedges.
        sub_topk: top-k neighbors retained per anchor item in substitute hyperedges.
        min_support: minimum pair support to keep complementary/substitute relation edges.
        sub_context_min: minimum transition count for a context item to contribute substitute pairs.
        conflict_margin: minimum normalized strength margin used to decide comp/sub conflicts.

    Returns:
        A tuple (comp_adj, sub_adj, comp_hyper, sub_hyper):
          - comp_adj/sub_adj: pairwise relation graphs.
          - comp_hyper/sub_hyper: item-level hypergraph propagation matrices.
    """
    comp_adj = dict()
    sub_adj = dict()
    comp_support = dict()
    sub_support = dict()
    item_freq = np.zeros(n_node, dtype=np.float32)
    prev_to_next = dict()
    next_to_prev = dict()
    dist_weights = [1.0 / d for d in range(1, max_gap + 1)]

    def context_pair_weight(count_a, count_b):
        return (count_a * count_b) / (count_a + count_b + EPSILON)

    for sess in all_sessions:
        # convert 1-indexed item ids to 0-indexed graph ids; skip padding/out-of-range ids
        seq = [x - 1 for x in sess if x != 0 and 1 <= x <= n_node]
        if len(seq) <= 1:
            continue
        for it in seq:
            item_freq[it] += 1.0

        # complementary: close-by ordered co-occurrence in the same session
        for i in range(len(seq)):
            src = seq[i]
            right = min(len(seq), i + 1 + max_gap)
            for j in range(i + 1, right):
                dst = seq[j]
                if src == dst:
                    continue
                w = dist_weights[j - i - 1]
                if src not in comp_adj:
                    comp_adj[src] = dict()
                if dst not in comp_adj:
                    comp_adj[dst] = dict()
                comp_adj[src][dst] = comp_adj[src].get(dst, 0.0) + w
                comp_adj[dst][src] = comp_adj[dst].get(src, 0.0) + w
                key_ab = _ordered_pair(src, dst)
                comp_support[key_ab] = comp_support.get(key_ab, 0) + 1

        # contexts for substitute learning
        for i in range(len(seq) - 1):
            p = seq[i]
            n = seq[i + 1]
            if p not in prev_to_next:
                prev_to_next[p] = dict()
            if n not in next_to_prev:
                next_to_prev[n] = dict()
            if n not in prev_to_next[p]:
                prev_to_next[p][n] = 0.0
            if p not in next_to_prev[n]:
                next_to_prev[n][p] = 0.0
            prev_to_next[p][n] += 1.0
            next_to_prev[n][p] += 1.0

    def add_sub_pair(i, j, w):
        if i == j:
            return
        if i not in sub_adj:
            sub_adj[i] = dict()
        sub_adj[i][j] = sub_adj[i].get(j, 0.0) + w

    # substitute: different items competing under same previous context
    for _, nxt_dict in prev_to_next.items():
        items = list(nxt_dict.items())
        for a in range(len(items)):
            ia, ca = items[a]
            if ca < sub_context_min:
                continue
            for b in range(a + 1, len(items)):
                ib, cb = items[b]
                if cb < sub_context_min:
                    continue
                w = context_pair_weight(ca, cb)
                add_sub_pair(ia, ib, w)
                add_sub_pair(ib, ia, w)
                key_ab = _ordered_pair(ia, ib)
                sub_support[key_ab] = sub_support.get(key_ab, 0) + int(min(ca, cb))

    # substitute: different items leading to same next context
    for _, prv_dict in next_to_prev.items():
        items = list(prv_dict.items())
        for a in range(len(items)):
            ia, ca = items[a]
            if ca < sub_context_min:
                continue
            for b in range(a + 1, len(items)):
                ib, cb = items[b]
                if cb < sub_context_min:
                    continue
                w = context_pair_weight(ca, cb)
                add_sub_pair(ia, ib, w)
                add_sub_pair(ib, ia, w)
                key_ab = _ordered_pair(ia, ib)
                sub_support[key_ab] = sub_support.get(key_ab, 0) + int(min(ca, cb))

    # prune weak relations
    for (ia, ib), support in list(comp_support.items()):
        if support < min_support:
            if ia in comp_adj and ib in comp_adj[ia]:
                del comp_adj[ia][ib]
            if ib in comp_adj and ia in comp_adj[ib]:
                del comp_adj[ib][ia]
            del comp_support[(ia, ib)]

    for (ia, ib), support in list(sub_support.items()):
        if support < min_support:
            if ia in sub_adj and ib in sub_adj[ia]:
                del sub_adj[ia][ib]
            if ib in sub_adj and ia in sub_adj[ib]:
                del sub_adj[ib][ia]
            del sub_support[(ia, ib)]

    # resolve comp/sub conflicts based on normalized strengths
    overlap_pairs = set(comp_support.keys()) & set(sub_support.keys())
    for ia, ib in overlap_pairs:
        norm = np.sqrt(max(item_freq[ia], EPSILON) * max(item_freq[ib], EPSILON))
        comp_strength = comp_support[(ia, ib)] / (norm + 1e-8)
        sub_strength = sub_support[(ia, ib)] / (norm + 1e-8)
        if comp_strength >= sub_strength + conflict_margin:
            if ia in sub_adj and ib in sub_adj[ia]:
                del sub_adj[ia][ib]
            if ib in sub_adj and ia in sub_adj[ib]:
                del sub_adj[ib][ia]
        elif sub_strength >= comp_strength + conflict_margin:
            if ia in comp_adj and ib in comp_adj[ia]:
                del comp_adj[ia][ib]
            if ib in comp_adj and ia in comp_adj[ib]:
                del comp_adj[ib][ia]

    comp = _dict_to_coo_with_self_loop(comp_adj, n_node, self_loop=1.0)
    sub = _dict_to_coo_with_self_loop(sub_adj, n_node, self_loop=1.0)
    comp_H = _build_anchor_item_hyperedges(comp_adj, n_node, topk=comp_topk, min_neighbors=1)
    sub_H = _build_anchor_item_hyperedges(sub_adj, n_node, topk=sub_topk, min_neighbors=1)
    comp_hyper = _incidence_to_hypergraph_propagation(comp_H, n_node)
    sub_hyper = _incidence_to_hypergraph_propagation(sub_H, n_node)
    return comp, sub, comp_hyper, sub_hyper

class Data():
    def __init__(self, data, all_train, shuffle=False, n_node=None, comp_sub_config=None):
        if isinstance(data, tuple):
            data = list(data)
        if len(data) == 2:
            self.raw_items = np.array(data[0], dtype=object)
            self.raw_events = None
            self.targets = np.asarray(data[1])
        else:
            self.raw_items = np.array(data[0], dtype=object)
            self.raw_events = np.array(data[1], dtype=object)
            self.targets = np.asarray(data[2])
        if isinstance(all_train, list) and len(all_train) == 2 and isinstance(all_train[0], (list, np.ndarray)):
            all_train_items = all_train[0]
            all_train_events = all_train[1]
        else:
            all_train_items = all_train
            all_train_events = None
        if comp_sub_config is None:
            comp_sub_config = {}
        max_gap = max(1, int(comp_sub_config.get('max_gap', COMP_SUB_DEFAULTS['max_gap'])))
        comp_topk = max(1, int(comp_sub_config.get('comp_topk', COMP_SUB_DEFAULTS['comp_topk'])))
        sub_topk = max(1, int(comp_sub_config.get('sub_topk', COMP_SUB_DEFAULTS['sub_topk'])))
        min_support = max(1, int(comp_sub_config.get('min_support', COMP_SUB_DEFAULTS['min_support'])))
        sub_context_min = max(1, int(comp_sub_config.get('sub_context_min', COMP_SUB_DEFAULTS['sub_context_min'])))
        conflict_margin = max(0.0, float(comp_sub_config.get('conflict_margin', COMP_SUB_DEFAULTS['conflict_margin'])))

        print(f"Building graphs with n_node={n_node}, max_gap={max_gap}, comp_topk={comp_topk}, sub_topk={sub_topk}, min_support={min_support}, sub_context_min={sub_context_min}, conflict_margin={conflict_margin}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            adj = data_masks(all_train_items, n_node)
            R = data_R(all_train_items, n_node)
            R1 = data_R1(all_train_items, n_node)
            adj_fuzzy = data_masks_fuzzy(all_train_items, n_node, all_events=all_train_events)
            R_fuzzy = data_R_fuzzy(all_train_items, n_node, all_events=all_train_events)
            R1_fuzzy = data_R1_fuzzy(all_train_items, n_node, all_events=all_train_events)
            comp_adj, sub_adj, comp_hyper, sub_hyper = data_item_hypergraph_comp_sub(
                all_train_items,
                n_node,
                max_gap=max_gap,
                comp_topk=comp_topk,
                sub_topk=sub_topk,
                min_support=min_support,
                sub_context_min=sub_context_min,
                conflict_margin=conflict_margin
            )
            self.adjacency = adj.multiply(1.0 / (adj.sum(axis=0).reshape(1, -1) + 1e-8))
            self.adjacency_T = self.adjacency.T
            self.adjacency1 = R1.multiply(1.0 / (R1.sum(axis=0).reshape(1, -1) + 1e-8))
            self.adjacency_comp = comp_adj.multiply(1.0 / (comp_adj.sum(axis=0).reshape(1, -1) + 1e-8))
            self.adjacency_sub = sub_adj.multiply(1.0 / (sub_adj.sum(axis=0).reshape(1, -1) + 1e-8))
            self.hyper_comp = comp_hyper
            self.hyper_sub = sub_hyper

            self.adjacency_fuzzy = adj_fuzzy.multiply(1.0 / (adj_fuzzy.sum(axis=0).reshape(1, -1) + 1e-8))
            self.adjacency_T_fuzzy = self.adjacency_fuzzy.T
            self.adjacency1_fuzzy = R1_fuzzy.multiply(1.0 / (R1_fuzzy.sum(axis=0).reshape(1, -1) + 1e-8))
            self.adj1 = adj.sum(axis=0).reshape(1, -1)
            self.adj2 = adj.sum(axis=1).reshape(1, -1)
            self.R = R.sum(axis=0).reshape(1, -1)
            self.R1 = R1.sum(axis=0).reshape(1, -1)
            self.adj1_fuzzy = adj_fuzzy.sum(axis=0).reshape(1, -1)
            self.adj2_fuzzy = adj_fuzzy.sum(axis=1).reshape(1, -1)
            self.R_fuzzy = R_fuzzy.sum(axis=0).reshape(1, -1)
            self.R1_fuzzy = R1_fuzzy.sum(axis=0).reshape(1, -1)
            self.comp_deg = comp_adj.sum(axis=0).reshape(1, -1)
            self.sub_deg = sub_adj.sum(axis=0).reshape(1, -1)
        self.n_node = n_node
        self.length = len(self.raw_items)
        self.shuffle = shuffle
        print(f"Graph construction complete. Raw data shape: {self.raw_items.shape}")
        print(f"Number of sessions: {self.length}")

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.raw_items = self.raw_items[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
            if self.raw_events is not None:
                self.raw_events = self.raw_events[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = []
        for i in range(n_batch):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, self.length)
            slices.append(np.arange(start_idx, end_idx))
        print(f"Generated {len(slices)} batches, batch_size={batch_size}")
        return slices

    def get_slice(self, index):
        if isinstance(index, (list, np.ndarray)):
            indices = np.array(index, dtype=int)
        else:
            indices = np.array([index], dtype=int)
        inp_items = self.raw_items[indices]
        inp_events = self.raw_events[indices] if self.raw_events is not None else None
        num_node = []
        for session in inp_items:
            if isinstance(session, np.ndarray):
                session = session.tolist()
            num_node.append(len([x for x in session if x != 0]))
        max_n_node = max(num_node) if num_node else 0
        session_len = []
        reversed_sess_item = []
        reversed_sess_event = []
        mask = []
        items_padded = []
        events_padded = []
        for idx, session in enumerate(inp_items):
            if isinstance(session, np.ndarray):
                session = session.tolist()
            events = None
            if inp_events is not None:
                events = inp_events[idx]
                if isinstance(events, np.ndarray):
                    events = events.tolist()
            nonzero_elems = [i for i, x in enumerate(session) if x != 0]
            session_len_val = len(nonzero_elems)
            session_len.append([session_len_val])
            padded_session = session + [0] * (max_n_node - session_len_val)
            items_padded.append(padded_session)
            mask.append([1] * session_len_val + [0] * (max_n_node - session_len_val))
            reversed_session = list(reversed(session[:session_len_val])) + [0] * (max_n_node - session_len_val)
            reversed_sess_item.append(reversed_session)
            if events is not None:
                padded_events = events + [0] * (max_n_node - session_len_val)
                events_padded.append(padded_events)
                reversed_event = list(reversed(events[:session_len_val])) + [0] * (max_n_node - session_len_val)
                reversed_sess_event.append(reversed_event)
        items_array = np.array(items_padded)
        mask_array = np.array(mask)
        reversed_array = np.array(reversed_sess_item)
        session_len_array = np.array(session_len)

        if inp_events is not None:
            events_array = np.array(events_padded)
            reversed_event_array = np.array(reversed_sess_event)
            return self.targets[indices] - 1, session_len_array, items_array, events_array, reversed_array, reversed_event_array, mask_array
        return self.targets[indices] - 1, session_len_array, items_array, reversed_array, mask_array
