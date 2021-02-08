import copy
import os
import pickle
import csv
import random
import time
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import argparse
from tools import *
from model import *

parser = argparse.ArgumentParser(description='gene')  # 创建parser对象
parser.add_argument('--lr', default=0.01, type=float, help='lr')
args = parser.parse_args()  # 解析参数，此处args是一个命名空间列表

submodular = True
if submodular:
    Model_Name = 'subRNN'
else:
    Model_Name = 'RNN'


def Training(arr, batch_size, n_epoch, lr):
    arr_Train, arr_Test = train_test_split(arr, test_size=0.1, random_state=6)
    arr_Train, arr_Validation = train_test_split(arr_Train, test_size=0.1, random_state=4)

    y_Train = arr_Train[:, 0]
    X_Train = preparation(arr_Train)

    y_Test = arr_Test[:, 0]
    X_Test = preparation(arr_Test)

    y_Validation = arr_Validation[:, 0]
    X_Validation = preparation(arr_Validation)

    output_file = '../outputs/gene/' + 'Normalized/' + Model_Name + '/' + str(lr) + '/'
    if os.path.isdir(output_file):
        pass
    else:
        os.mkdir(output_file)

    log_f = open(
        '../Logs/gene/TEST_%s_%s.bak' % (
            Model_Name, lr), 'w+')
    print('constructing the optimizer ...', file=log_f, flush=True)

    rnn = RNN()
    if torch.cuda.is_available():
        rnn = rnn.cuda()

    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(rnn.parameters(), lr=lr, momentum=0.9)
    print('done!', file=log_f, flush=True)
    # define cross entropy loss function

    # weights = torch.FloatTensor([5, 1, 10])
    # BCEloss = torch.nn.BCEWithLogitsLoss(weights).cuda()
    CEloss = torch.nn.CrossEntropyLoss().cuda()

    n_batches = int(np.ceil(float(len(X_Train)) / float(batch_size)))

    print('training start', file=log_f, flush=True)

    rnn.train()

    best_train_cost = 0.0
    best_validate_cost = 100000000.0
    epoch_duaration = 0.0
    best_epoch = 0.0

    for epoch in range(n_epoch):
        iteration = 0
        cost_vector = []
        start_time = time.time()
        samples = random.sample(range(n_batches), n_batches)

        for index in samples:
            batch_diagnosis_codes = X_Train[batch_size * index: batch_size * (index + 1)]
            batch_labels = y_Train[batch_size * index: batch_size * (index + 1)]
            t_diagnosis_codes, t_labels = pad_matrix(batch_diagnosis_codes, batch_labels, 5)

            model_input = copy.deepcopy(t_diagnosis_codes)
            for i in range(len(model_input)):
                for j in range(len(model_input[i])):
                    idx = 0
                    for k in range(len(model_input[i][j])):
                        model_input[i][j][k] = idx
                        idx += 1

            model_input = torch.FloatTensor(model_input).cuda()
            t_labels = torch.LongTensor(t_labels).cuda()

            optimizer.zero_grad()

            logit = rnn(model_input, torch.tensor(t_diagnosis_codes).cuda())

            pred = torch.argmax(logit, 1)
            # pred = torch.max(logit, 1)[1].view((1,)).data.numpy()
            one_hot_t_labels = one_hot_labels(t_labels, n_lables)
            one_hot_t_labels = torch.FloatTensor(one_hot_t_labels).cuda()

            loss = CEloss(logit, t_labels)
            loss.backward()

            optimizer.step()

            if submodular:
                for p in rnn.parameters():
                    p.data += 0.1
                    p.data = abs(p.data)
                    p.data -= 0.1

            cost_vector.append(loss.cpu().data.numpy())

            # if iteration % 60 == 0:
            #     print('epoch:%d, iteration:%d/%d, cost:%f' % (epoch, iteration, n_batches, loss.cpu().data.numpy()),
            #           file=log_f, flush=True)
            #     # print(pred)
            iteration += 1

        duration = time.time() - start_time
        train_cost = np.mean(cost_vector)
        validate_cost = calculate_cost(rnn, X_Validation, y_Validation, batch_size)
        epoch_duaration += duration

        if validate_cost < best_validate_cost:
            # torch.save(rnn.state_dict(), output_file + 'Adam_' + Model_Name + '.' + str(epoch))
            torch.save(rnn.state_dict(), output_file + 'Adam_' + Model_Name + '.' + str(epoch),
                       _use_new_zipfile_serialization=False)
        print('epoch:%d, mean_cost:%f, duration:%f' % (epoch, np.mean(cost_vector), duration), file=log_f, flush=True)

        if validate_cost < best_validate_cost:
            best_validate_cost = validate_cost
            best_train_cost = train_cost
            best_epoch = epoch

        buf = 'Best Epoch:%d, Train_Cost:%f, Valid_Cost:%f' % (best_epoch, best_train_cost, best_validate_cost)
        print(buf, file=log_f, flush=True)
        print()

    # test

    print('-----------test--------------', file=log_f, flush=True)
    best_parameters_file = output_file + 'Adam_' + Model_Name + '.' + str(best_epoch)


    print(best_parameters_file)
    rnn.load_state_dict(torch.load(best_parameters_file))
    rnn.eval()
    batch_size = 16
    n_batches = int(np.ceil(float(len(X_Train)) / float(batch_size)))
    y_true = np.array([])
    y_pred = np.array([])

    for index in range(n_batches):  # n_batches

        batch_diagnosis_codes = X_Train[batch_size * index: batch_size * (index + 1)]
        batch_labels = y_Train[batch_size * index: batch_size * (index + 1)]
        t_diagnosis_codes, t_labels = pad_matrix(batch_diagnosis_codes, batch_labels, 5)

        model_input = copy.copy(t_diagnosis_codes)
        for i in range(len(model_input)):
            for j in range(len(model_input[i])):
                idx = 0
                for k in range(len(model_input[i][j])):
                    model_input[i][j][k] = idx
                    idx += 1

        model_input = torch.FloatTensor(model_input).cuda()

        logit = rnn(model_input, torch.tensor(t_diagnosis_codes).cuda())

        prediction = torch.max(logit, 1)[1].view((len(t_labels),)).data.cpu().numpy()

        y_true = np.concatenate((y_true, t_labels))
        y_pred = np.concatenate((y_pred, prediction))

    accuary = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average='macro')

    print('Training data')
    print('accuary:, precision:, recall:, f1:', (accuary, precision, recall, f1))

    log_a = open(
        '../Logs/gene/TEST____%s_Adam_%s.bak' % (
            Model_Name, lr), 'w+')
    print(best_parameters_file, file=log_a, flush=True)
    print('Training data', file=log_a, flush=True)
    print('accuary:, precision:, recall:, f1:', (accuary, precision, recall, f1), file=log_a, flush=True)
    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.zeros((5, 3))
    n_batches_test = int(np.ceil(float(len(X_Test)) / float(batch_size)))
    for index in range(n_batches_test):  # n_batches

        batch_diagnosis_codes = X_Test[batch_size * index: batch_size * (index + 1)]
        batch_labels = y_Test[batch_size * index: batch_size * (index + 1)]
        t_diagnosis_codes, t_labels = pad_matrix(batch_diagnosis_codes, batch_labels, 5)

        model_input = copy.copy(t_diagnosis_codes)
        for i in range(len(model_input)):
            for j in range(len(model_input[i])):
                idx = 0
                for k in range(len(model_input[i][j])):
                    model_input[i][j][k] = idx
                    idx += 1

        model_input = torch.FloatTensor(model_input).cuda()

        logit = rnn(model_input, torch.tensor(t_diagnosis_codes).cuda())

        prediction = torch.max(logit, 1)[1].view((len(t_labels),)).data.cpu().numpy()
        logit = logit.data.cpu().numpy()

        y_true = np.concatenate((y_true, t_labels))
        y_pred = np.concatenate((y_pred, prediction))
        y_score = np.concatenate((y_score, logit))

    accuary = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average='macro')
    print('Testing data')
    print('accuary:, precision:, recall:, f1:', (accuary, precision, recall, f1))

    print('Testing data', file=log_a, flush=True)
    print('accuary:, precision:, recall:, f1:', (accuary, precision, recall, f1), file=log_a, flush=True)

    return 0


arr = load_csv_data('./Dataset/splice.csv')

batch_size = 32
n_epoch = 15000
lr = args.lr
n_lables = 3

rnn = RNN()
if torch.cuda.is_available():
    rnn = rnn.cuda()

print('num_layers =' + str(rnn.num_layers))
print('hidden_size = ', rnn.hidden_size)
print('lr =' + str(lr))

Training(arr, batch_size, n_epoch, lr)



