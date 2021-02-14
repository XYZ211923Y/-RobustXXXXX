import numpy as np
import torch
import copy


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


def multi_perturb(data, pf_valid = 0.2, pf_change = 1/1104):
    """
    Randomly flip bits.

    Parameters
    ----------
    data: torch.Tensor [batch_size, num_features]
        The indices of the non-zero elements
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one

    Returns
    -------
    data_perturbed: torch.Tensor [batch_size, num_features]
        The indices of the non-zero elements after perturbation
    """
    data = data.cuda()
    valid = torch.LongTensor(data.shape).bernoulli_(pf_valid).cuda()
    changed_to = np.argmax(np.random.multinomial(1, [pf_change]*1104, size=data.shape), axis=1)
    changed_to = torch.tensor(changed_to).cuda()
    data_perturbed = data * (1-valid) + valid * changed_to
    return data_perturbed


def calculate_cost(model, X, y, batch_size):
    n_batches = int(np.ceil(float(len(X)) / float(batch_size)))
    cost_sum = 0.0
    weights = torch.FloatTensor([2, 2, 1])
    BCEloss = torch.nn.BCEWithLogitsLoss(weights).cuda()
    for index in range(n_batches):
        batch_diagnosis_codes = X[batch_size * index: batch_size * (index + 1)]
        batch_labels = y[batch_size * index: batch_size * (index + 1)]
        t_diagnosis_codes, t_labels = pad_matrix(batch_diagnosis_codes, batch_labels, 1104)

        model_input = copy.copy(t_diagnosis_codes)
        for i in range(len(model_input)):
            for j in range(len(model_input[i])):
                idx = 0
                for k in range(len(model_input[i][j])):
                    model_input[i][j][k] = idx
                    idx += 1

        model_input = torch.FloatTensor(model_input).cuda()
        t_labels = torch.LongTensor(t_labels).cuda()

        one_hot_t_labels = one_hot_labels(t_labels, 3)
        one_hot_t_labels = torch.FloatTensor(one_hot_t_labels).cuda()

        logit = model(model_input, torch.tensor(t_diagnosis_codes).cuda())
        loss = BCEloss(logit, one_hot_t_labels)
        cost_sum += loss.cpu().data.numpy()
    return cost_sum / n_batches