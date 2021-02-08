import numpy as np
import torch
import copy
import torch.nn.functional as F
import csv


def pad_matrix(seq_diagnosis_codes, seq_labels, n_diagnosis_codes):
    lengths = np.array([len(seq) for seq in seq_diagnosis_codes])
    n_samples = len(seq_diagnosis_codes)
    maxlen = np.max(lengths)

    f_1 = 1e-5
    batch_diagnosis_codes = f_1 * np.ones((maxlen, n_samples, n_diagnosis_codes), dtype=np.float32)

    for idx, c in enumerate(seq_diagnosis_codes):
        for x, subseq in zip(batch_diagnosis_codes[:, idx, :], c[:]):
            l = 1
            f_2 = float((l - f_1 * (n_diagnosis_codes - l)) / l)
            x[subseq] = f_2

    batch_labels = np.array(seq_labels, dtype=np.int64)

    return batch_diagnosis_codes, batch_labels


def one_hot_labels(t_labels, n_labels):
    one_hot = np.zeros((len(t_labels), n_labels), dtype=np.int64)
    for index in range(len(t_labels)):
        one_hot[index][t_labels[index]] = 1
    return one_hot


def calculate_cost(model, X, y, batch_size):
    n_batches = int(np.ceil(float(len(X)) / float(batch_size)))
    cost_sum = 0.0
    for index in range(n_batches):
        batch_diagnosis_codes = X[batch_size * index: batch_size * (index + 1)]
        batch_labels = y[batch_size * index: batch_size * (index + 1)]
        t_diagnosis_codes, t_labels = pad_matrix(batch_diagnosis_codes, batch_labels, 5)

        model_input = copy.copy(t_diagnosis_codes)
        for i in range(len(model_input)):
            for j in range(len(model_input[i])):
                idx = 0
                for k in range(len(model_input[i][j])):
                    model_input[i][j][k] = idx
                    idx += 1

        model_input = torch.FloatTensor(model_input).cuda()
        t_labels = torch.LongTensor(t_labels).cuda()

        logit = model(model_input, torch.tensor(t_diagnosis_codes).cuda())
        loss = F.cross_entropy(logit, t_labels)
        cost_sum += loss.cpu().data.numpy()
    return cost_sum / n_batches


def randSelect_multicate(discreteData, p_remain=0.4, p_change=0.2, ra=2):
    discreteData = np.array(discreteData)
    data = copy.deepcopy(discreteData)

    zero_idx = np.where(discreteData == 0)[0]
    zero_select = np.argmax(np.random.multinomial(50, [p_remain, p_change, p_change, p_change], size=len(zero_idx)),
                            axis=1)
    add_idx = np.random.choice(np.where(zero_select != 0)[0], size=min(ra, len(np.where(zero_select != 0)[0])),
                               replace=False, p=None)
    data[zero_idx[add_idx]] = zero_select[add_idx]

    one_idx = np.where(discreteData == 1)[0]
    one_select = np.argmax(np.random.multinomial(50, [p_change, p_remain, p_change, p_change], size=len(one_idx)),
                           axis=1)
    add_idx = np.random.choice(np.where(one_select != 1)[0], size=min(ra, len(np.where(one_select != 1)[0])),
                               replace=False, p=None)
    data[one_idx[add_idx]] = one_select[add_idx]

    two_idx = np.where(discreteData == 2)[0]
    two_select = np.argmax(np.random.multinomial(50, [p_change, p_change, p_remain, p_change], size=len(two_idx)),
                           axis=1)
    add_idx = np.random.choice(np.where(two_select != 2)[0], size=min(ra, len(np.where(two_select != 2)[0])),
                               replace=False, p=None)
    data[two_idx[add_idx]] = two_select[add_idx]

    three_idx = np.where(discreteData == 3)[0]
    three_select = np.argmax(np.random.multinomial(50, [p_change, p_change, p_change, p_remain], size=len(three_idx)),
                             axis=1)
    add_idx = np.random.choice(np.where(three_select != 3)[0], size=min(ra, len(np.where(three_select != 3)[0])),
                               replace=False, p=None)
    data[three_idx[add_idx]] = three_select[add_idx]

    four_idx = np.where(discreteData == 4)[0]
    four_select = np.argmax(np.random.multinomial(50, [p_change, p_change, p_change, p_change, p_change],
                                                  size=len(four_idx)), axis=1)
    add_idx = np.random.choice(np.where(four_select != 4)[0], size=min(ra, len(np.where(four_select != 4)[0])),
                               replace=False, p=None)
    data[four_idx[add_idx]] = four_select[add_idx]

    return data


def preparation(data):
    X = data[:, 2]
    X = X.tolist()
    X_temp = []
    for i in range(len(X)):
        for j in range(len(X[0])):
            X_temp.append(X[i][j])
    X_temp = np.array(X_temp, dtype=int)
    X_temp = X_temp.reshape((len(X), len(X[0])))
    X_temp = torch.from_numpy(X_temp)
    X = X_temp
    return X


def load_csv_data(csvfile):
    f = open(csvfile, 'rb')
    reader = csv.reader(f)
    result = list(reader)
    for i in range(len(result)):
        for j in range(3):
            result[i][j] = result[i][j].strip()
    A = 0
    G = 1
    T = 2
    C = 3
    N = 4

    for i in range(len(result)):
        num_feature = []
        if result[i][0] == 'IE':
            result[i][0] = 0
        elif result[i][0] == 'N':
            result[i][0] = 1
        else:
            result[i][0] = 2
        for j in range(60):
            if result[i][2][j] == 'A':
                num_feature.append(A)
            elif result[i][2][j] == 'G':
                num_feature.append(G)
            elif result[i][2][j] == 'T':
                num_feature.append(T)
            elif result[i][2][j] == 'C':
                num_feature.append(C)
            else:
                num_feature.append(N)
        result[i][2] = num_feature

    arr = np.array(result)
    return arr
