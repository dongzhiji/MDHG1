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

opt = parser.parse_args()
# 设置日志文件
log_file = setup_logging()
logging.info(f"日志文件: {log_file}")
logging.info(f"运行参数: {opt}")

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

    train_data = Data(train_data, all_train, shuffle=False, n_node=n_node)
    test_data = Data(test_data, all_train, shuffle=False, n_node=n_node)

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
        comp_sub_pair_hyper_mix=opt.comp_sub_pair_hyper_mix
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