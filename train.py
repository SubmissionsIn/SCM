import torch
from network import Network
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
from scipy.spatial import distance
from torch.utils.data import DataLoader
from metric import valid
from sklearn.metrics import confusion_matrix
import time
from TSNE import TSNE_PLOT as ttsne
# MNIST-USPS  (aka. DIGIT)
# BDGP
# Fashion
# NGs
# VOC
# WebKB
# DHA
# Fc_COIL_20 (aka. COIL-20)

# SCM_w/o_DA
# SCM_w/o_NoiseDA
# SCM_w/o_MaskDA
# SCM
# SCM_REC
# SCM_REC_ETC
# SCM_ETC

Dataname = 'MNIST-USPS'
MODE = 'SCM'
miss_rate = 0.25
noise_rate = 0.25
Gaussian_noise = 0.4
tsne = True  # True / False
T = 1

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_iterations", default=200)
parser.add_argument("--con_iterations", default=50)
parser.add_argument("--tune_iterations", default=50)
parser.add_argument("--feature_dim", default=256)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument('--mode', type=str, default=MODE)
parser.add_argument('--miss_rate', type=str, default=miss_rate)
parser.add_argument('--noise_rate', type=str, default=noise_rate)
parser.add_argument('--Gaussian_noise', type=str, default=Gaussian_noise)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if args.dataset == "MNIST-USPS":
    args.con_iterations = 1000
    args.mse_iterations = 1000
    args.gamma = 0.02
    args.alpha = 0.5
    args.beta = 0.5
    seed = 1
if args.dataset == "BDGP":
    args.con_iterations = 400
    args.mse_iterations = 3000
    args.gamma = 0.1
    args.alpha = 0.5
    args.beta = 0.5
    seed = 4
if args.dataset == "Fashion":
    args.con_iterations = 20000
    args.mse_iterations = 2500
    args.gamma = 0.003
    args.alpha = 0.2
    args.beta = 0.81
    seed = 1
if args.dataset == "DHA":
    args.con_iterations = 500
    args.mse_iterations = 700
    args.gamma = 0.02
    args.alpha = 0.2
    args.beta = 0.5
    seed = 4
if args.dataset == "WebKB":
    args.con_iterations = 200
    args.mse_iterations = 200
    args.gamma = 0.001
    args.alpha = 0.6
    args.beta = 0.6
    seed = 2
if args.dataset == "NGs":
    args.con_iterations = 200
    args.mse_iterations = 800
    args.gamma = 0.00005
    args.alpha = 0.5
    args.beta = 0.5
    seed = 5
if args.dataset == "VOC":
    args.con_iterations = 200
    args.mse_iterations = 900
    args.gamma = 0.002
    args.alpha = 0.01
    args.beta = 0.37
    seed = 9
if args.dataset == "Fc_COIL_20":
    args.con_iterations = 2000
    args.mse_iterations = 400
    args.gamma = 0.031
    args.alpha = 0.2
    args.beta = 0.5
    seed = 1

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True
def mask(rows, cols, p):
    tensor = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        if i < int(rows * p):
            while True:
                row = np.random.randint(0, 2, size=cols)
                if np.count_nonzero(row) < cols and np.count_nonzero(row) > 0:
                    tensor[i, :] = row
                    break
        else:
            tensor[i, :] = 1
    np.random.shuffle(tensor)
    tensor = torch.tensor(tensor)
    return tensor
def add_noise(matrix, std, p):
    rows, cols = matrix.shape
    noisy_matrix = matrix.clone()
    for i in range(rows):
        if random.random() < p:
            noise = torch.randn(cols, device=device) * std
            noisy_matrix[i] += noise
    return noisy_matrix
def scale_normalize_matrix(input_matrix, min_value=0, max_value=1):
    min_val = input_matrix.min()
    max_val = input_matrix.max()
    input_range = max_val - min_val
    scaled_matrix = (input_matrix - min_val) / input_range * (max_value - min_value) + min_value

    return scaled_matrix
dataset, _, view, data_size, class_num, dimss = load_data(args.dataset)
data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )
def SCM(iteration, mode,miss_rate,noise_rate,Gaussian_noise):
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, y, _) in enumerate(data_loader):
        # print(y)        # different batches
        for v in range(view):
            xs[v] = xs[v].to(device)
        break
    masked_xs = []
    noised_xs = []
    num_rows = xs[0].shape[0]
    mask_tensor = mask(num_rows,view,miss_rate).to(device)
    for v in range(view):
        masked_x = mask_tensor[:,v].unsqueeze(1)*xs[v]
        masked_xs.append(masked_x)
    for v in range(view):
        noised_x = add_noise(xs[v],Gaussian_noise,noise_rate)
        noised_xs.append(noised_x)
    xs_all = torch.cat(xs,dim=1)
    mask_all = torch.cat(masked_xs, dim=1)
    noise_all = torch.cat(noised_xs, dim=1)
    optimizer.zero_grad()
    xrs,_,xs_z,q = model(xs_all)
    mask_xrs,_,mask_z,_ = model(mask_all)
    noise_xrs,_,noise_z,_ = model(noise_all)
    loss_xrs = mse(xs_all,xrs)
    loss_mask = mse(xs_all,mask_xrs)
    loss_noise = mse(xs_all,noise_xrs)
    if mode =='SCM' or mode == 'SCM_REC'or mode =='SCM_REC_ETC'or mode =='SCM_ETC':
        loss_con_1 = criterion.forward_feature(noise_z, mask_z)
        loss_con_2 = criterion.forward_feature(mask_z, noise_z)
    if mode =='SCM_w/o_MaskDA':
        loss_con_1 = criterion.forward_feature(noise_z, xs_z)
        loss_con_2 = criterion.forward_feature(xs_z, noise_z)
    if mode == 'SCM_w/o_NoiseDA':
        loss_con_1 = criterion.forward_feature(mask_z, xs_z)
        loss_con_2 = criterion.forward_feature(xs_z, mask_z)
    if mode == 'SCM_w/o_DA':
        loss_con_1 = criterion.forward_feature(xs_z, xs_z)
        loss_con_2 = criterion.forward_feature(xs_z, xs_z)
    if mode =='SCM_REC_ETC' or mode == 'SCM_REC':
        loss = loss_xrs + loss_mask + loss_noise + loss_con_1 + loss_con_2
    if mode == 'SCM' or mode == 'SCM_ETC' or mode =='SCM_w/o_NoiseDA' or mode =='SCM_w/o_DA' or mode =='SCM_w/o_MaskDA':
        loss = loss_con_1+loss_con_2
    loss.backward()
    optimizer.step()
    print('Iteration {}'.format(iteration), 'Loss:{:.6f}'.format(loss))
def destiny_peak(model, device, gamma=args.gamma, alpha=args.alpha, beta=args.beta, metric='euclidean'):
    ALL_loader = DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    for step, (xs, ys, _) in enumerate(ALL_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        xs_all = torch.cat(xs, dim=1)
        with torch.no_grad():
            _, _, z, _ = model.forward(xs_all)
            z = z.cpu().detach().numpy()

    condensed_distance = distance.pdist(z, metric=metric)
    d_c = np.sort(condensed_distance)[int(len(condensed_distance) * gamma)]
    redundant_distance = distance.squareform(condensed_distance)
    rho = np.sum(np.exp(-(redundant_distance / d_c) ** 2), axis=1)
    order_distance = np.argsort(redundant_distance, axis=1)
    delta = np.zeros_like(rho)
    nn = np.zeros_like(rho).astype(int)
    for i in range(len(delta)):
        mask = rho[order_distance[i]] > rho[i]
        if mask.sum() > 0:
            nn[i] = order_distance[i][mask][0]
            delta[i] = redundant_distance[i, nn[i]]
        else:
            nn[i] = order_distance[i, -1]
            delta[i] = redundant_distance[i, nn[i]]
    rho_c = min(rho) + (max(rho) - min(rho)) * alpha
    delta_c = min(delta) + (max(delta) - min(delta)) * beta
    centers = np.where(np.logical_and(rho > rho_c, delta > delta_c))[0]
    num_clusters = len(centers)
    cluster_points = z[centers]
    probabilities = np.zeros((z.shape[0], num_clusters))
    for i in range(z.shape[0]):
        for j in range(num_clusters):
            probabilities[i, j] = np.exp(-np.linalg.norm(z[i] - cluster_points[j]))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    yyy = torch.from_numpy(probabilities)
    yyy = torch.argmax(yyy, dim=1)
    confusion = confusion_matrix(yyy, ys)
    per = np.sum(np.max(confusion, axis=0)) / np.sum(confusion)
    additional_columns = 64 - probabilities.shape[1]
    zero_columns = np.zeros((probabilities.shape[0], additional_columns))
    probabilities = np.hstack((probabilities, zero_columns))
    probabilities = torch.from_numpy(probabilities)
    print('num:{}'.format(num_clusters), 'accuracy:{:.6f}'.format(per))
    return probabilities
def end2end(iteration,probability_matrix,mode,miss_rate,noise_rate,Gaussian_noise):

    if iteration > args.mse_iterations:
        mse = torch.nn.MSELoss()
        masked_xs = []
        noised_xs = []
        for batch_idx, (xs, _, idx) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
                idx[v] = idx[v].to(device)
        num_rows = xs[0].shape[0]
        mask_tensor = mask(num_rows , view, miss_rate).to(device)
        for v in range(view):
            masked_x = mask_tensor[:, v].unsqueeze(1) * xs[v]
            masked_xs.append(masked_x)
        for v in range(view):
            noised_x = add_noise(xs[v], Gaussian_noise, noise_rate)
            noised_xs.append(noised_x)
        xs_all = torch.cat(xs, dim=1)
        mask_all = torch.cat(masked_xs, dim=1)
        noise_all = torch.cat(noised_xs, dim=1)
        optimizer.zero_grad()
        xrs, _, z_all, q = model(xs_all)
        mask_xrs, _, mask_z, mask_q  = model(mask_all)
        noise_xrs, _, noise_z, noise_q = model(noise_all)
        select_rows = probability_matrix[idx]
        qs = np.vstack(select_rows)
        qs = torch.from_numpy(qs).float()
        qs = qs.to(device)
        qs = scale_normalize_matrix(qs)
        loss_xrs = mse(xs_all, xrs)
        loss_mask = mse(xs_all, mask_xrs)
        loss_noise = mse(xs_all, noise_xrs)
        loss_con_1 = criterion.forward_feature(noise_z, mask_z)
        loss_con_2 = criterion.forward_feature(mask_z, noise_z)
        loss_con = mse(qs, q)
        if mode == 'SCM_REC_ETC':
            loss = loss_xrs + loss_mask + loss_noise + loss_con_1 + loss_con_2 + loss_con
        if mode == 'SCM_ETC':
            loss =  loss_con_1 + loss_con_2 + loss_con
        loss.backward()
        optimizer.step()
        print('Epoch {}'.format(iteration), 'Loss:{:.6f}'.format(loss))
accs = []
nmis = []
purs = []
aris = []

for i in range(T):
    print("ROUND:{}".format(i + 1))
    setup_seed(seed)
    model = Network(dimss, args.feature_dim, args.high_feature_dim,device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, device).to(device)
    mode = args.mode
    miss_rate = args.miss_rate
    noise_rate = args.noise_rate
    Gaussian_noise = args.Gaussian_noise

    time0 = time.time()
    iteration = 1
    while iteration <= args.con_iterations:
        SCM(iteration,mode,miss_rate,noise_rate,Gaussian_noise)
        if iteration == args.con_iterations:
            if mode == 'SCM' or mode =='SCM_REC' or mode =='SCM_w/o_NoiseDA' or mode =='SCM_w/o_DA' or mode =='SCM_w/o_MaskDA':
                acc, nmi, ari, pur = valid(model, device, dataset, view, data_size, class_num, eval_z=True)
                accs.append(acc)
                nmis.append(nmi)
                purs.append(pur)
                aris.append(ari)
        iteration += 1

    if mode == 'SCM_ETC' or mode =='SCM_REC_ETC':
        probability_matrix = destiny_peak(model, device)
        while iteration <= args.mse_iterations + args.con_iterations:
            end2end(iteration, probability_matrix,mode,miss_rate,noise_rate,Gaussian_noise)
            if iteration == args.mse_iterations + args.con_iterations:
                acc, nmi, ari, pur = valid(model, device, dataset, view, data_size, class_num, eval_q =True)
                accs.append(acc)
                nmis.append(nmi)
                purs.append(pur)
                aris.append(ari)
            iteration += 1


print('%.4f'% np.mean(accs), '%.4f'% np.std(accs), accs)
print('%.4f'% np.mean(nmis), '%.4f'% np.std(nmis), nmis)
print('%.4f'% np.mean(aris), '%.4f'% np.std(aris), aris)


if tsne == True:
    miss_x = []
    noise_x = []
    model.eval()
    ALL_loader = DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    for step, (xs, ys, _) in enumerate(ALL_loader):
        ys = ys.numpy()
        for v in range(view):
            xs[v] = xs[v].to(device)
        num_rows = xs[0].shape[0]
        miss =  mask(num_rows,view,miss_rate).to(device)
        for v in range(view):
            miss = miss[:,v].unsqueeze(1)*xs[v]
            miss_x.append(miss)
        for v in range(view):
            noisedx = add_noise(xs[v],Gaussian_noise,noise_rate)
            noise_x.append(noisedx)

        xs_all = torch.cat(xs, dim=1)
        mask_xx = torch.cat(miss_x, dim=1)
        noise_xx = torch.cat(noise_x, dim=1)
        with torch.no_grad():
            _, _, z, _ = model.forward(xs_all)
            _, _, zm, _ = model.forward(mask_xx)
            _, _, zn, _ = model.forward(noise_xx)
            z = z.cpu().detach().numpy()
            zm = zm.cpu().detach().numpy()
            zn = zn.cpu().detach().numpy()

        ttsne(z, ys, "z")
        ttsne(zm, ys, "zm")
        ttsne(zn, ys, "zn")
