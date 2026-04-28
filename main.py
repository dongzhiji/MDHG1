import time
from util import Data
from model import *
import os
import argparse
import pickle
import logging
from datetime import datetime
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

def resolve_dataset_dir(dataset_name):
    default_dir = os.path.join('datasets', dataset_name)
    if os.path.isdir(default_dir):
        return default_dir
    lower_dir = os.path.join('datasets', dataset_name.lower())
    if os.path.isdir(lower_dir):
        return lower_dir
    try:
        for entry in os.listdir('datasets'):
            if entry.lower() == dataset_name.lower():
                return os.path.join('datasets', entry)
    except OSError:
        pass
    return default_dir

def load_pickle_with_fallback(primary_path, fallback_paths=()):
    tried = []
    for path in [primary_path] + list(fallback_paths):
        tried.append(path)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f), path
    raise FileNotFoundError(f"Missing dataset file. Tried: {', '.join(tried)}")
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

def main():
    logging.info("=" * 60)
    logging.info("开始加载数据...")
    dataset_key = opt.dataset.lower()
    dataset_dir = resolve_dataset_dir(opt.dataset)
    train_path = os.path.join(dataset_dir, 'train.txt')
    test_path = os.path.join(dataset_dir, 'test.txt')
    all_train_path = os.path.join(dataset_dir, 'all_train_seq.txt')

    train_data, train_src = load_pickle_with_fallback(train_path, [all_train_path, test_path])
    test_data, test_src = load_pickle_with_fallback(test_path, [train_path])
    all_train, all_train_src = load_pickle_with_fallback(all_train_path, [train_path, test_path])

    if train_src != train_path:
        logging.warning(f"训练数据缺失，使用替代文件: {train_src}")
    if test_src != test_path:
        logging.warning(f"测试数据缺失，使用替代文件: {test_src}")
    if all_train_src != all_train_path:
        logging.warning(f"全量训练序列缺失，使用替代文件: {all_train_src}")

    if dataset_key == 'tmall':
        n_node = 40727
    elif dataset_key == 'retailrocket':#最大的 item ID = 36968
        n_node = 36968
    elif dataset_key == 'amazon':
        n_node = 18888
    elif dataset_key == 'lastfm':#最大的 item ID = 38997
        n_node = 38997
    elif dataset_key == 'diginetica':#最大的 item ID = 43097
        n_node =43097
    elif dataset_key == 'nowplaying':
        n_node = 60416
    else:
        n_node = 309
    logging.info(f"数据集: {opt.dataset} (目录: {dataset_dir}), 节点数: {n_node}")
    comp_sub_cache_dir = opt.comp_sub_cache_dir if opt.comp_sub_cache_dir else os.path.join(dataset_dir, 'graph_cache')

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
        cache_prefix=f"{opt.dataset}_train",
        dataset=dataset_key
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
        cache_prefix=f"{opt.dataset}_train",
        dataset=dataset_key
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
        comp_sub_decouple_weight=opt.comp_sub_decouple_weight
    ))

    #reset_parameters(model)
    logging.info("模型创建完成，参数初始化完成")

    top_K = [5, 10, 20, 50]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]

    logging.info(f"开始训练，共 {opt.epoch} 个epoch")

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

    # 最终结果汇总
    logging.info("=" * 60)
    logging.info("训练完成！最终最佳结果:")
    for K in top_K:
        logging.info(
            f'K={K}: Hit={best_results["metric%d" % K][0]:.4f}%, MRR={best_results["metric%d" % K][1]:.4f}%, NDCG={best_results["metric%d" % K][2]:.4f}%')

    # 保存结果到文件
    result_file = log_file.replace('.log', '_results.txt')
    with open(result_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"数据集: {opt.dataset}\n")
        f.write(f"参数: {opt}\n")
        f.write("=" * 60 + "\n")
        f.write("最终最佳结果:\n")
        for K in top_K:
            f.write(
                f"K={K}: Hit={best_results['metric%d' % K][0]:.4f}%, MRR={best_results['metric%d' % K][1]:.4f}%, NDCG={best_results['metric%d' % K][2]:.4f}%\n")
        f.write("=" * 60 + "\n")

    logging.info(f"结果已保存到: {result_file}")


if __name__ == '__main__':
    start_time = time.time()
    init_seed(2026)
    main()
    end_time = time.time()
    logging.info(f"总运行时间: {end_time - start_time:.2f} 秒")
