import time
from util import Data
from model import *
import os
import argparse
import pickle
import logging
from datetime import datetime
HIT_IDX = 0
MRR_IDX = 1
NDCG_IDX = 2
# 设置日志
def setup_logging():
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    # 配置logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='dataset name: retailrocket/diginetica/Nowplaying/sample')
parser.add_argument('--epoch', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--embSize', type=int, default=128, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--layer', type=int, default=2, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.005, help='ssl task maginitude')
parser.add_argument('--lam', type=float, default=0.0001, help='diff task maginitude')
parser.add_argument('--eps', type=float, default=0.2, help='eps')
parser.add_argument('--K1', type=int, default=80, help='numbers')
parser.add_argument('--K2', type=int, default=50, help='numbers')
parser.add_argument('--K3', type=int, default=20, help='numbers')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu_id')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--intent_align_weight', type=float, default=0.03, help='weight of intent-item alignment loss')
parser.add_argument('--short_intent_min', type=float, default=0.10, help='minimum short-term intent fusion gate')
parser.add_argument('--short_intent_max', type=float, default=0.45, help='maximum short-term intent fusion gate')
parser.add_argument('--short_len_factor_min', type=float, default=0.35, help='minimum session-length factor in short-term intent fusion')
parser.add_argument('--comp_sub_pair_hyper_mix', type=float, default=0.5, help='blend ratio for pairwise vs hypergraph comp/sub embeddings')
parser.add_argument('--comp_max_gap', type=int, default=3, help='max sequence distance for complementary relation mining')
parser.add_argument('--comp_sub_topk', type=int, default=8, help='top-k related items in each comp/sub hyperedge')
parser.add_argument('--comp_sub_min_neighbors', type=int, default=1, help='minimum neighbors required to form a comp/sub hyperedge')
parser.add_argument('--comp_sub_min_support', type=float, default=1.0, help='minimum raw relation support before normalization')
parser.add_argument('--comp_sub_min_norm_weight', type=float, default=0.02, help='minimum normalized relation weight kept in comp/sub graph')
parser.add_argument('--sub_context_topk', type=int, default=20, help='max candidates per substitute context (prev/next)')
parser.add_argument('--sub_context_min', type=int, default=2, help='minimum candidates per substitute context')
parser.add_argument('--comp_symmetric', type=int, default=1, help='whether complementary relation graph is symmetric (1) or directed (0)')
parser.add_argument('--comp_sub_cache', type=int, default=1, help='enable cache for item-level comp/sub relation mining')
parser.add_argument('--comp_sub_cache_dir', default='', help='cache directory for mined comp/sub relation graphs')
parser.add_argument('--sub_co_buy_suppress', type=float, default=0.6, help='suppression strength for substitute pairs with strong direct co-buy')
parser.add_argument('--comp_head_quantile', type=float, default=0.8, help='head item frequency quantile for complementary relation bucket threshold')
parser.add_argument('--comp_head_scale', type=float, default=1.15, help='threshold scale for complementary head bucket')
parser.add_argument('--comp_tail_scale', type=float, default=0.85, help='threshold scale for complementary tail bucket')
parser.add_argument('--sub_head_quantile', type=float, default=0.8, help='head item frequency quantile for substitute relation bucket threshold')
parser.add_argument('--sub_head_scale', type=float, default=1.15, help='threshold scale for substitute head bucket')
parser.add_argument('--sub_tail_scale', type=float, default=0.85, help='threshold scale for substitute tail bucket')
parser.add_argument('--comp_sub_decouple_weight', type=float, default=0.02, help='regularization weight for comp/sub embedding decorrelation')
parser.add_argument('--logit_comp_scale', type=float, default=0.20, help='session-aware comp-logit residual scale')
parser.add_argument('--logit_sub_scale', type=float, default=0.25, help='session-aware sub-logit residual scale')
parser.add_argument('--logit_short_sub_boost', type=float, default=0.30, help='short-intent boost on substitute logit residual')
parser.add_argument('--comp_sub_warmup_epochs', type=int, default=1, help='warmup epochs before enabling comp/sub relation branches')
parser.add_argument('--comp_sub_ramp_epochs', type=int, default=4, help='ramp epochs to reach full comp/sub branch strength')
parser.add_argument('--rel_conf_comp_scale', type=float, default=1.0, help='reliability logit scale for complementary branch')
parser.add_argument('--rel_conf_sub_scale', type=float, default=1.0, help='reliability logit scale for substitute branch')
parser.add_argument('--rel_conf_event_gain', type=float, default=0.20, help='event-strength gain for relation reliability')
parser.add_argument('--rel_conf_repeat_penalty', type=float, default=0.25, help='repeat-ratio penalty for relation reliability')
parser.add_argument('--rel_conf_len_gain', type=float, default=0.15, help='session-length gain for relation reliability')
parser.add_argument('--enable_comp_branch', type=int, default=1, help='enable complementary hypergraph branch (1/0)')
parser.add_argument('--enable_sub_branch', type=int, default=1, help='enable substitute hypergraph branch (1/0)')
parser.add_argument('--enable_logit_residual', type=int, default=1, help='enable comp/sub residual logit fusion (1/0)')
parser.add_argument('--enable_rel_conf_gate', type=int, default=1, help='enable relation reliability gating (1/0)')
parser.add_argument('--early_stop_patience', type=int, default=0, help='early stop patience on monitor metric; 0 disables early stop')
parser.add_argument('--early_stop_metric', default='mrr10', choices=['mrr10', 'ndcg10', 'hit10'], help='metric used for early stopping')
parser.add_argument('--early_stop_min_epoch', type=int, default=5, help='minimum epochs before early-stop check starts')
parser.add_argument('--seed', type=int, default=2026, help='random seed')
parser.add_argument('--seed_list', default='', help='comma-separated seeds, e.g., "2026,2027,2028"; empty uses --seed')
parser.add_argument('--amp', type=int, default=0, help='deprecated: AMP is disabled and this flag is ignored')

opt = parser.parse_args()
# 设置日志文件
log_file = setup_logging()
logging.info(f"日志文件: {log_file}")
logging.info(f"运行参数: {opt}")
if opt.amp:
    logging.warning("参数 --amp 已废弃且会被忽略，当前固定不使用AMP。")

if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)
    torch.cuda.set_device(0)
    logging.info(f"CUDA is available. Using GPU {opt.gpu_id}")
    logging.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    logging.info("CUDA is not available. Running on CPU.")
    opt.gpu_id = -1

def reset_parameters(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Embedding):
            nn.init.xavier_uniform_(layer.weight)

def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to: {seed}")

def parse_seed_list(seed_list_str, default_seed):
    if seed_list_str is None or seed_list_str.strip() == '':
        return [int(default_seed)]
    seeds = []
    for token in seed_list_str.split(','):
        token = token.strip()
        if token == '':
            continue
        try:
            seeds.append(int(token))
        except ValueError as e:
            raise ValueError(
                f"Invalid seed value '{token}' in --seed_list='{seed_list_str}', expected comma-separated integers"
            ) from e
    return seeds if len(seeds) > 0 else [int(default_seed)]

def run_single_seed(seed):
    logging.info("=" * 60)
    logging.info(f"开始加载数据... seed={seed}")
    init_seed(seed)

    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))
    all_train = pickle.load(open('datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))

    if opt.dataset == 'Tmall':
        n_node = 40727
    elif opt.dataset == 'retailrocket':#最大的 item ID = 36968
        n_node = 36968
    elif opt.dataset == 'amazon':
        n_node = 18888
    elif opt.dataset == 'lastfm':#最大的 item ID = 38997
        n_node = 38997
    elif opt.dataset == 'diginetica':#最大的 item ID = 43097
        n_node =43097
    else:
        n_node = 309
    logging.info(f"数据集: {opt.dataset}, 节点数: {n_node}")
    comp_sub_cache_dir = opt.comp_sub_cache_dir if opt.comp_sub_cache_dir else os.path.join('datasets', opt.dataset, 'graph_cache')

    train_data = Data(
        train_data, all_train, shuffle=False, n_node=n_node, comp_max_gap=opt.comp_max_gap,
        comp_sub_topk=opt.comp_sub_topk, comp_sub_min_neighbors=opt.comp_sub_min_neighbors,
        comp_sub_min_support=opt.comp_sub_min_support, comp_sub_min_norm_weight=opt.comp_sub_min_norm_weight,
        sub_context_topk=opt.sub_context_topk, sub_context_min=opt.sub_context_min,
        comp_symmetric=bool(opt.comp_symmetric),
        sub_co_buy_suppress=opt.sub_co_buy_suppress,
        comp_head_quantile=opt.comp_head_quantile, comp_head_scale=opt.comp_head_scale, comp_tail_scale=opt.comp_tail_scale,
        sub_head_quantile=opt.sub_head_quantile, sub_head_scale=opt.sub_head_scale, sub_tail_scale=opt.sub_tail_scale,
        comp_sub_cache=bool(opt.comp_sub_cache),
        comp_sub_cache_dir=comp_sub_cache_dir,
        cache_prefix=f"{opt.dataset}_train"
    )
    test_data = Data(
        test_data, all_train, shuffle=False, n_node=n_node, comp_max_gap=opt.comp_max_gap,
        comp_sub_topk=opt.comp_sub_topk, comp_sub_min_neighbors=opt.comp_sub_min_neighbors,
        comp_sub_min_support=opt.comp_sub_min_support, comp_sub_min_norm_weight=opt.comp_sub_min_norm_weight,
        sub_context_topk=opt.sub_context_topk, sub_context_min=opt.sub_context_min,
        comp_symmetric=bool(opt.comp_symmetric),
        sub_co_buy_suppress=opt.sub_co_buy_suppress,
        comp_head_quantile=opt.comp_head_quantile, comp_head_scale=opt.comp_head_scale, comp_tail_scale=opt.comp_tail_scale,
        sub_head_quantile=opt.sub_head_quantile, sub_head_scale=opt.sub_head_scale, sub_tail_scale=opt.sub_tail_scale,
        comp_sub_cache=bool(opt.comp_sub_cache),
        comp_sub_cache_dir=comp_sub_cache_dir,
        cache_prefix=f"{opt.dataset}_train"
    )

    logging.info("创建模型...")
    model = trans_to_cuda(MDHG(
        R=train_data.R,
        adj1=train_data.adj1,
        adj2=train_data.adj2,
        adjacency=train_data.adjacency,
        adjacency_T=train_data.adjacency_T,
        adjacency1=train_data.adjacency1,
        adjacency_comp=train_data.adjacency_comp,
        adjacency_sub=train_data.adjacency_sub,
        hyper_comp=train_data.hyper_comp,
        hyper_sub=train_data.hyper_sub,
        R1=train_data.R1,
        comp_deg=train_data.comp_deg,
        sub_deg=train_data.sub_deg,
        adjacency_fuzzy=train_data.adjacency_fuzzy,
        adjacency_T_fuzzy=train_data.adjacency_T_fuzzy,
        adjacency1_fuzzy=train_data.adjacency1_fuzzy,
        adj1_fuzzy=train_data.adj1_fuzzy,
        adj2_fuzzy=train_data.adj2_fuzzy,
        R_fuzzy=train_data.R_fuzzy,
        R1_fuzzy=train_data.R1_fuzzy,
        n_node=n_node,
        lr=opt.lr,
        l2=opt.l2,
        beta=opt.beta,
        lam=opt.lam,
        eps=opt.eps,
        layers=opt.layer,
        emb_size=opt.embSize,
        batch_size=opt.batchSize,
        dataset=opt.dataset,
        K1=opt.K1,
        K2=opt.K2,
        K3=opt.K3,
        dropout=opt.dropout,
        alpha=opt.alpha,
        intent_align_weight=opt.intent_align_weight,
        short_intent_min=opt.short_intent_min,
        short_intent_max=opt.short_intent_max,
        short_len_factor_min=opt.short_len_factor_min,
        comp_sub_pair_hyper_mix=opt.comp_sub_pair_hyper_mix,
        comp_sub_decouple_weight=opt.comp_sub_decouple_weight,
        logit_comp_scale=opt.logit_comp_scale,
        logit_sub_scale=opt.logit_sub_scale,
        logit_short_sub_boost=opt.logit_short_sub_boost,
        comp_sub_warmup_epochs=opt.comp_sub_warmup_epochs,
        comp_sub_ramp_epochs=opt.comp_sub_ramp_epochs,
        rel_conf_comp_scale=opt.rel_conf_comp_scale,
        rel_conf_sub_scale=opt.rel_conf_sub_scale,
        rel_conf_event_gain=opt.rel_conf_event_gain,
        rel_conf_repeat_penalty=opt.rel_conf_repeat_penalty,
        rel_conf_len_gain=opt.rel_conf_len_gain,
        enable_comp_branch=bool(opt.enable_comp_branch),
        enable_sub_branch=bool(opt.enable_sub_branch),
        enable_logit_residual=bool(opt.enable_logit_residual),
        enable_rel_conf_gate=bool(opt.enable_rel_conf_gate)
    ))

    #reset_parameters(model)
    logging.info("模型创建完成，参数初始化完成")

    top_K = [5, 10, 20, 50]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]

    logging.info(f"开始训练，共 {opt.epoch} 个epoch")

    best_monitor = -1e9
    early_stop_bad_count = 0
    epochs_run = 0
    for epoch in range(opt.epoch):
        logging.info('-' * 60)
        logging.info(f'Epoch: {epoch}')

        metrics, total_loss = train_test(model, train_data, test_data, epoch)

        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            metrics['ndcg%d' % K] = np.mean(metrics['ndcg%d' % K]) * 100

            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
            if best_results['metric%d' % K][2] < metrics['ndcg%d' % K]:
                best_results['metric%d' % K][2] = metrics['ndcg%d' % K]
                best_results['epoch%d' % K][2] = epoch

        # 记录当前epoch的指标
        logging.info(f'Epoch {epoch} 结果:')
        for K in top_K:
            logging.info(
                f'  Hit@{K}: {metrics["hit%d" % K]:.4f}%, MRR@{K}: {metrics["mrr%d" % K]:.4f}%, NDCG@{K}: {metrics["ndcg%d" % K]:.4f}%')
        logging.info(f'  Loss: {total_loss:.4f}')

        # 记录最佳结果
        logging.info(f'当前最佳结果:')
        for K in top_K:
            logging.info(
                f'  Best Hit@{K}: {best_results["metric%d" % K][0]:.4f}% (epoch {best_results["epoch%d" % K][0]}), '
                f'Best MRR@{K}: {best_results["metric%d" % K][1]:.4f}% (epoch {best_results["epoch%d" % K][1]}), '
                f'Best NDCG@{K}: {best_results["metric%d" % K][2]:.4f}% (epoch {best_results["epoch%d" % K][2]})')
        monitor_score = metrics[opt.early_stop_metric]
        if monitor_score > best_monitor + 1e-8:
            best_monitor = monitor_score
            early_stop_bad_count = 0
        else:
            early_stop_bad_count += 1
        epochs_run = epoch + 1
        if (
            opt.early_stop_patience > 0 and
            epochs_run >= max(1, opt.early_stop_min_epoch) and
            early_stop_bad_count >= opt.early_stop_patience
        ):
            logging.info(
                f"触发早停: metric={opt.early_stop_metric}, patience={opt.early_stop_patience}, "
                f"best={best_monitor:.4f}, stop_epoch={epoch}"
            )
            break

    return best_results, epochs_run

def main():
    seed_list = parse_seed_list(opt.seed_list, opt.seed)
    logging.info(f"统一评估协议: seed_list={seed_list}, early_stop_metric={opt.early_stop_metric}, "
                 f"patience={opt.early_stop_patience}, min_epoch={opt.early_stop_min_epoch}")
    all_seed_results = []
    for seed in seed_list:
        best_results, epochs_run = run_single_seed(seed)
        all_seed_results.append((seed, best_results, epochs_run))

    logging.info("=" * 60)
    logging.info("训练完成！最终最佳结果:")
    for seed, best_results, epochs_run in all_seed_results:
        logging.info(f"[seed={seed}] epochs_run={epochs_run}")
        for K in [5, 10, 20, 50]:
            logging.info(
                f'  K={K}: Hit={best_results["metric%d" % K][0]:.4f}%, '
                f'MRR={best_results["metric%d" % K][1]:.4f}%, '
                f'NDCG={best_results["metric%d" % K][2]:.4f}%'
            )

    result_file = log_file.replace('.log', '_results.txt')
    with open(result_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"数据集: {opt.dataset}\n")
        f.write(f"参数: {opt}\n")
        f.write(f"seed_list: {seed_list}\n")
        f.write("=" * 60 + "\n")
        for seed, best_results, epochs_run in all_seed_results:
            f.write(f"[seed={seed}] epochs_run={epochs_run}\n")
            for K in [5, 10, 20, 50]:
                f.write(
                    f"K={K}: Hit={best_results['metric%d' % K][0]:.4f}%, "
                    f"MRR={best_results['metric%d' % K][1]:.4f}%, "
                    f"NDCG={best_results['metric%d' % K][2]:.4f}%\n"
                )
            f.write("-" * 60 + "\n")
        if len(all_seed_results) > 1:
            mrr10 = [r[1]['metric10'][MRR_IDX] for r in all_seed_results]
            ndcg10 = [r[1]['metric10'][NDCG_IDX] for r in all_seed_results]
            hit10 = [r[1]['metric10'][HIT_IDX] for r in all_seed_results]
            f.write(
                f"seed-avg@10: Hit={np.mean(hit10):.4f}%, "
                f"MRR={np.mean(mrr10):.4f}%, NDCG={np.mean(ndcg10):.4f}%\n"
            )
            f.write(
                f"seed-std@10: Hit={np.std(hit10):.4f}, "
                f"MRR={np.std(mrr10):.4f}, NDCG={np.std(ndcg10):.4f}\n"
            )
        f.write("=" * 60 + "\n")

    logging.info(f"结果已保存到: {result_file}")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    logging.info(f"总运行时间: {end_time - start_time:.2f} 秒")
