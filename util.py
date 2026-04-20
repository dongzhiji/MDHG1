import numpy as np
from scipy.sparse import coo_matrix
import warnings


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


def _relation_fuzzy_weight(freq, pos_i, sess_len, e_i, e_j, relation='r1', repeat_ratio=0.0):
    # 1) 频次隶属度
    if relation == 'r3':
        mu_freq = fuzzy_membership(freq, center=2.0, scale=8.0)
    else:
        mu_freq = fuzzy_membership(freq, center=1.0, scale=5.0)

    # 2) 位置/时序隶属度（越靠近当前位置越高）
    pos_center = max((sess_len - 1) / 2.0, 0.0)
    mu_time = fuzzy_membership(pos_i, center=pos_center, scale=max(sess_len / 3.0, 1.0))

    # 3) 行为强度隶属度
    beh = (_event_strength(e_i) + _event_strength(e_j)) / 2.0
    if relation == 'r1':
        mu_beh = fuzzy_membership(beh, center=1.0, scale=1.0)
    elif relation == 'r2':
        mu_beh = fuzzy_membership(beh, center=0.8, scale=1.2)
    else:  # r3 共现
        mu_beh = fuzzy_membership(beh, center=1.2, scale=1.0)

    # 4) 一致性（重复率越高，顺序关系可信度降低；共现略提升）
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


def data_item_hypergraph_comp_sub(all_sessions, n_node, max_gap=3):
    """
    Build item-view hypergraph relations learned from sessions.

    Args:
        all_sessions: list of session item sequences.
        n_node: total number of items.
        max_gap: maximum distance window for complementary relation extraction.

    Returns:
        A tuple (comp_adj, sub_adj) where each entry is a scipy COO sparse matrix
        with shape [n_node, n_node]:
          - comp_adj: complementary item relation graph.
          - sub_adj: substitute item relation graph.
    """
    comp_adj = dict()
    sub_adj = dict()
    prev_to_next = dict()
    next_to_prev = dict()
    dist_weights = [1.0 / d for d in range(1, max_gap + 1)]

    def context_pair_weight(count_a, count_b):
        return (count_a * count_b) / (count_a + count_b + 1e-8)

    for sess in all_sessions:
        seq = [x - 1 for x in sess if x != 0 and 1 <= x <= n_node]
        if len(seq) <= 1:
            continue

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
            for b in range(a + 1, len(items)):
                ib, cb = items[b]
                w = context_pair_weight(ca, cb)
                add_sub_pair(ia, ib, w)
                add_sub_pair(ib, ia, w)

    # substitute: different items leading to same next context
    for _, prv_dict in next_to_prev.items():
        items = list(prv_dict.items())
        for a in range(len(items)):
            ia, ca = items[a]
            for b in range(a + 1, len(items)):
                ib, cb = items[b]
                w = context_pair_weight(ca, cb)
                add_sub_pair(ia, ib, w)
                add_sub_pair(ib, ia, w)

    comp = _dict_to_coo_with_self_loop(comp_adj, n_node, self_loop=1.0)
    sub = _dict_to_coo_with_self_loop(sub_adj, n_node, self_loop=1.0)
    return comp, sub


class Data():
    def __init__(self, data, all_train, shuffle=False, n_node=None):
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

        print(f"Building graphs with n_node={n_node}...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            adj = data_masks(all_train_items, n_node)
            R = data_R(all_train_items, n_node)
            R1 = data_R1(all_train_items, n_node)

            adj_fuzzy = data_masks_fuzzy(all_train_items, n_node, all_events=all_train_events)
            R_fuzzy = data_R_fuzzy(all_train_items, n_node, all_events=all_train_events)
            R1_fuzzy = data_R1_fuzzy(all_train_items, n_node, all_events=all_train_events)
            comp_adj, sub_adj = data_item_hypergraph_comp_sub(all_train_items, n_node)

            self.adjacency = adj.multiply(1.0 / (adj.sum(axis=0).reshape(1, -1) + 1e-8))
            self.adjacency_T = self.adjacency.T
            self.adjacency1 = R1.multiply(1.0 / (R1.sum(axis=0).reshape(1, -1) + 1e-8))
            self.adjacency_comp = comp_adj.multiply(1.0 / (comp_adj.sum(axis=0).reshape(1, -1) + 1e-8))
            self.adjacency_sub = sub_adj.multiply(1.0 / (sub_adj.sum(axis=0).reshape(1, -1) + 1e-8))

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

    # 其余函数 generate_batch/get_slice 保持你原版不变

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
