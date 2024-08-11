from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch

def scale_normalize_matrix(input_matrix, min_value=0, max_value=1):
    min_val = input_matrix.min()
    max_val = input_matrix.max()
    input_range = max_val - min_val
    scaled_matrix = (input_matrix - min_val) / input_range * (max_value - min_value) + min_value
    return scaled_matrix

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur

def inference(loader, model, device, view):
    model.eval()
    soft_vector = []

    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        xs_all = torch.cat(xs, dim=1)
        with torch.no_grad():
            _, h, z, q = model.forward(xs_all)
        z = z.cpu().detach().numpy()
        h = h.cpu().detach().numpy()
        q = q.detach()
        soft_vector.extend(q.cpu().detach().numpy())
    total_pred = np.argmax(np.array(soft_vector), axis=1)

    y = y.numpy()
    y = y.flatten()
    return y, h, z, total_pred
def valid(model, device, dataset, view, data_size, class_num, eval_q = False,eval_z = False):
    test_loader = DataLoader(
            dataset,
            batch_size=data_size,
            shuffle=False,
        )
    labels_vector, h, z, q = inference(test_loader, model, device, view)
    kmeans = KMeans(n_clusters=class_num)
    print(str(len(labels_vector)) + " samples")
    if eval_q == True:
        nmi_q, ari_q, acc_q, pur_q = evaluate(labels_vector, q)
        print('ACC_q = {:.4f} NMI_q = {:.4f} ARI_q = {:.4f} PUR_q = {:.4f}'.format(acc_q, nmi_q, ari_q, pur_q))
        return acc_q, nmi_q, ari_q, pur_q
    if eval_z == True:
        z_pred = kmeans.fit_predict(z)
        nmi_z, ari_z, acc_z, pur_z = evaluate(labels_vector, z_pred)
        print('ACC_z = {:.4f} NMI_z = {:.4f} ARI_z = {:.4f} PUR_z = {:.4f}'.format(acc_z, nmi_z, ari_z, pur_z))
        return acc_z, nmi_z, ari_z, pur_z
