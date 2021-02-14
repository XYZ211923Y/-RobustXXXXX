import pickle
import time
from itertools import combinations
from tools import *
from model import *


class Attacker(object):
    def __init__(self, best_parameters_file, log_f, emb_weights):
        self.n_diagnosis_codes = 1104
        self.n_labels = n_lables

        self.rnn = RNN(emb_weights)
        if torch.cuda.is_available():
            self.rnn = self.rnn.cuda()
        self.rnn.load_state_dict(torch.load(best_parameters_file, map_location='cpu'))
        self.rnn.eval()

        self.log_f = log_f

        self.alpha = 100

        self.criterion = nn.CrossEntropyLoss()

    def input_handle(self, funccall, y):  # input:funccall, output:(seq_len,n_sample,m)[i][j][k]=k,
        funccall = [funccall]
        y = [y]
        t_diagnosis_codes, _ = pad_matrix(funccall, y, self.n_diagnosis_codes)
        model_input = copy.deepcopy(t_diagnosis_codes)
        for i in range(len(model_input)):
            for j in range(len(model_input[i])):
                idx = 0
                for k in range(len(model_input[i][j])):
                    model_input[i][j][k] = idx
                    idx += 1

        model_input = torch.FloatTensor(model_input).cuda()
        return model_input, torch.tensor(t_diagnosis_codes).cuda()

    def classify(self, funccall, y):
        model_input, weight_of_embed_codes = self.input_handle(funccall, y)
        logit = self.rnn(model_input, weight_of_embed_codes)
        logit = logit.cpu()
        pred = torch.max(logit, 1)[1].view((1,)).data.numpy()

        logit = logit.data.cpu().numpy()
        label_set = {0, 1, 2}
        label_set.remove(y)
        list_label_set = list(label_set)
        g = logit[0][y]
        h1 = logit[0][list_label_set[0]]
        h2 = logit[0][list_label_set[1]]
        h = max(h1, h2)

        return pred, g, h, h1, h2

    def eval_object(self, eval_funccall, greedy_set, orig_label, current_h1, current_h2, current_g, changed_set,
                    greedy_set_visit_idx, greedy_set_best_temp_funccall, failure_set, K):
        candidate_lists = []
        success_flag = 1
        funccall_lists = []
        sets_lists = []
        label_set = {0, 1, 2}
        label_set.remove(orig_label)
        list_label_set = list(label_set)
        flip_set = set()
        flip_funccall = torch.tensor([])
        
        for i in range(0, len(greedy_set) + 1):
            subset1 = combinations(greedy_set, i)
            for subset in subset1:
                candidate_lists.append(list(subset))
                sets_lists.append(set(subset))

        for can in candidate_lists:

            temp_funccall = copy.deepcopy(eval_funccall)

            for position in can:
                visit_idx = position[0]
                code_idx = position[1]
                temp_funccall[visit_idx] = code_idx

            funccall_lists.append(temp_funccall)

        batch_size = 16
        n_batches = int(np.ceil(float(len(funccall_lists)) / float(batch_size)))
        self.rnn.train()
        grad_feature_list = torch.zeros((20, 1))
        grad_cate_index_list = torch.zeros((20, 1))
        max_grad = torch.tensor(0).float()
        max_subsets_object = -1
        max_subset_index = -1
        for index in range(n_batches):  # n_batches

            batch_diagnosis_codes = funccall_lists[batch_size * index: batch_size * (index + 1)]
            batch_labels = [orig_label] * len(batch_diagnosis_codes)
            t_diagnosis_codes, t_labels = pad_matrix(batch_diagnosis_codes, batch_labels, self.n_diagnosis_codes)

            model_input = copy.copy(t_diagnosis_codes)
            for i in range(len(model_input)):
                for j in range(len(model_input[i])):
                    idx = 0
                    for k in range(len(model_input[i][j])):
                        model_input[i][j][k] = idx
                        idx += 1

            model_input = torch.FloatTensor(model_input).cuda()

            t_diagnosis_codes = torch.tensor(t_diagnosis_codes).cuda()
            t_diagnosis_codes = torch.autograd.Variable(t_diagnosis_codes.data, requires_grad=True)

            logit = self.rnn(model_input, t_diagnosis_codes)
            loss = self.criterion(logit, torch.LongTensor(batch_labels).cuda())
            loss.backward()
            logit = logit.data.cpu().numpy()

            subsets_g = logit[:, orig_label]
            subsets_h = np.max([logit[:, list_label_set[0]], logit[:, list_label_set[1]]], axis=0)
            subsets_objects = subsets_h - subsets_g
            
            max_subset_object_temp = max(subsets_objects)
            if max_subset_object_temp > max_subsets_object:
                max_subsets_object = max_subset_object_temp
                max_subset_index = batch_size * index + np.argmax(subsets_objects)

            grad = t_diagnosis_codes.grad.cpu().data
            grad = torch.abs(grad)

            subsets_g = subsets_g.reshape(-1, 1)
            subsets_g = torch.tensor(subsets_g).transpose(0, 1)
            grad_feature_temp = torch.max(grad, dim=2)[0]
            grad_feature_temp = grad_feature_temp / subsets_g
            grad_cate_index = torch.argmax(grad, dim=2)
            max_grad = max(max_grad, torch.max(grad_feature_temp))

            if index == 0:
                grad_feature_list = grad_feature_temp
                grad_cate_index_list = grad_cate_index
            else:
                grad_feature_list = torch.cat((grad_feature_list, grad_feature_temp), dim=1)
                grad_cate_index_list = torch.cat((grad_cate_index_list, grad_cate_index), dim=1)

        if max_subsets_object >= 0:
            success_flag = 0
            flip_funccall = funccall_lists[max_subset_index]
            flip_set = sets_lists[max_subset_index]

            return max_subsets_object, greedy_set_best_temp_funccall, success_flag, changed_set, greedy_set, \
                   greedy_set_visit_idx, failure_set, flip_set, flip_funccall
        
        self.rnn.eval()
        grad_feature, grad_set_index_list = torch.max(grad_feature_list, dim=1)
        print("max_grad:", max_grad.item())
        funccalls = []
        features = []
        for index in range(len(grad_feature)):
            if index in greedy_set_visit_idx:
                continue
            if index in failure_set:
                continue
            temp_funccall = copy.deepcopy(funccall_lists[grad_set_index_list[index]])
            temp_funccall[index] = grad_cate_index_list[index, grad_set_index_list[index]]
            features.append(index)
            funccalls.append(temp_funccall)

        chosen_object = 0
        chosen_feature = 0

        if len(funccalls) == 0:
            success_flag = -2
            greedy_set_best_temp_funccall = funccall_lists[max_subset_index]
            changed_set = sets_lists[max_subset_index]

            return chosen_object, greedy_set_best_temp_funccall, success_flag, changed_set, greedy_set, \
                   greedy_set_visit_idx, failure_set, flip_set, flip_funccall
        temp_labels = [orig_label] * len(funccalls)
        t_diagnosis_codes, t_labels = pad_matrix(funccalls, temp_labels, self.n_diagnosis_codes)

        model_input = copy.copy(t_diagnosis_codes)
        for i in range(len(model_input)):
            for j in range(len(model_input[i])):
                id = 0
                for k in range(len(model_input[i][j])):
                    model_input[i][j][k] = id
                    id += 1

        model_input = torch.FloatTensor(model_input).cuda()

        t_diagnosis_codes = torch.tensor(t_diagnosis_codes).cuda()
        logit = self.rnn(model_input, t_diagnosis_codes)
        logit = logit.data.cpu().numpy()

        g = logit[:, orig_label]
        h1 = logit[:, list_label_set[0]]
        h2 = logit[:, list_label_set[1]]
        h = np.max([h1, h2], axis=0)
        objects = h - g
        for item in range(len(objects)):
            if objects[item] > 0:
                failure_set.add(features[item])

        functions = self.alpha * np.max([h1 - current_h1, h2 - current_h2], axis=0) - g + current_g

        sorted_feature, sorted_feature_index = torch.sort(torch.tensor(functions))

        real_k = min(K, len(features))
        while real_k > 0:
            idx = np.random.choice(real_k, 1)[0]
            chosen_feature_temp = sorted_feature_index[idx]
            chosen_object = objects[chosen_feature_temp]
            chosen_feature = features[chosen_feature_temp]
            if chosen_feature in failure_set:
                real_k = idx
                continue
            chosen_set = grad_set_index_list[chosen_feature]
            chosen_category = grad_cate_index_list[chosen_feature, chosen_set].item()
            changed_set = copy.deepcopy(sets_lists[chosen_set])
            if chosen_object < max_subsets_object:
                chosen_set = max_subset_index
                chosen_object = max_subsets_object
                chosen_funccall = copy.deepcopy(funccall_lists[chosen_set])
            else:
                chosen_funccall = copy.deepcopy(funccall_lists[chosen_set])
                chosen_funccall[chosen_feature] = chosen_category
                changed_set.add((chosen_feature, chosen_category))

            greedy_set_visit_idx.add(chosen_feature)
            greedy_set.add((chosen_feature, chosen_category))
            greedy_set_best_temp_funccall = chosen_funccall
            break

        if real_k == 0:
            success_flag = 0
            min_feature_temp = sorted_feature_index[0]
            min_feature = features[min_feature_temp]
            min_set = grad_set_index_list[min_feature]
            min_category = grad_cate_index_list[min_feature, min_set].item()
            flip_funccall = copy.deepcopy(funccall_lists[min_set])
            flip_funccall[chosen_feature] = min_category
            flip_set = copy.deepcopy(sets_lists[min_set])
            flip_set.add((min_feature, min_category))
            

        return chosen_object, greedy_set_best_temp_funccall, success_flag, changed_set, greedy_set, \
               greedy_set_visit_idx, failure_set, flip_set, flip_funccall

    def attack(self, funccall, y, failure_set):
        print()
        success_flag = 1

        orig_pred, orig_g, orig_h, orig_h1, orig_h2 = self.classify(funccall, y)

        greedy_set = set()
        greedy_set_visit_idx = set()
        greedy_set_best_temp_funccall = funccall
        changed_set = set()
        flip_set = set()

        mf_process = []
        greedy_set_process = []
        changed_set_process = []

        mf_process.append(np.float(orig_h - orig_g))

        n_changed = 0
        iteration = 0
        robust_flag = 0

        current_object = orig_h - orig_g
        current_g = orig_g
        current_h1 = orig_h1
        current_h2 = orig_h2
        flip_object = -1
        flip_funccall = funccall

        if current_object > 0:
            robust_flag = -1
            print("Original classification error")
            return mf_process, greedy_set_process, changed_set_process, \
                   iteration, robust_flag, np.float(orig_g), greedy_set, greedy_set_visit_idx, \
                   greedy_set_best_temp_funccall.tolist(), np.float(orig_g), np.float(current_object), changed_set, \
                   n_changed, flip_funccall.tolist(), flip_set, np.float(flip_object), failure_set

        print(current_object)
        k = 10
        while success_flag == 1:
            iteration += 1

            worst_object, greedy_set_best_temp_funccall, success_flag, changed_set, greedy_set, greedy_set_visit_idx, \
            failure_set, flip_set, flip_funccall = self.eval_object(funccall, greedy_set, y, current_h1, current_h2,
                                                                    current_g, changed_set, greedy_set_visit_idx,
                                                                    greedy_set_best_temp_funccall, failure_set, k)

            print(iteration)
            print(worst_object)
            print(greedy_set)

            changed_set_process.append(copy.deepcopy(changed_set))
            pred, g, h, h1, h2 = self.classify(greedy_set_best_temp_funccall, y)
            mf_process.append(np.float(h - g))
            greedy_set_process.append(copy.deepcopy(greedy_set))

            if iteration == 10:
                success_flag = -1
                robust_flag = 1

        for i in range(len(greedy_set_best_temp_funccall)):
            if greedy_set_best_temp_funccall[i] != funccall[i]:
                n_changed += 1
        print(greedy_set_best_temp_funccall)

        pred, g, h, h1, h2 = self.classify(greedy_set_best_temp_funccall, y)
        current_object = h - g
        print("Flip_set:", flip_set)
        if success_flag == 0:
            flip_pred, flip_g, flip_h, flip_h1, flip_h2 = self.classify(flip_funccall, y)
            flip_object = flip_h - flip_g
            mf_process.append(np.float(flip_h - flip_g))

        if success_flag == -2:
            flip_object = 0
            mf_process.append(np.float(0))

        print(flip_funccall)

        return mf_process, greedy_set_process, changed_set_process, \
               iteration, robust_flag, np.float(orig_g), greedy_set, greedy_set_visit_idx, \
               greedy_set_best_temp_funccall.tolist(), np.float(g), np.float(current_object), changed_set, n_changed, \
               flip_funccall.tolist(), flip_set, np.float(flip_object), failure_set

Model_Type = 'Normal'

print(Model_Type)

data_file = '../Logs/malware/%s/mindiff_original_funccall.pickle' % Model_Type

emb = torch.load("./Dataset/PretrainedEmbedding1104.0", map_location=torch.device('cpu'))['embeddings.weight']

emb_weights = emb.clone().detach()

test = pickle.load(open(data_file, 'rb'))
test = np.array(test)

n_lables = 3

best_parameters_file = ''
if Model_Type == 'Normal':
    best_parameters_file = './Classifier/Mal_RNN.942'
elif Model_Type == 'Submodular':
    best_parameters_file = './Classifier/Mal_subRNN.2'

X = test

mf_process_all = []
greedy_set_process_all = []
changed_set_process_all = []

iterations_all = []
robust_flag_all = []

orignal_funccalls_all = []
orignal_labels_all = []
orignal_g_all = []

final_greedy_set_all = []
final_greedy_set_visit_idx_all = []
final_funccall_all = []
final_g_all = []
final_mf_all = []
final_changed_set_all = []
final_changed_num_all = []

flip_funccall_all = []
flip_set_all = []
flip_mf_all = []
flip_sample_original_label_all = []
flip_sample_index_all = []

failue_set_all = []

log_attack = open(
    '../Logs/malware/%s/gradmin_Attack.bak' % Model_Type, 'w+')
attacker = Attacker(best_parameters_file, log_attack, emb_weights)
index = -1

for i in range(len(X)):
    print("---------------------- %d --------------------" % i, file=log_attack, flush=True)

    sample = X[i]
    label, _, _, _, _ = attacker.classify(sample, 0)
    label = np.int(label)

    failure_set = set()

    print('* Processing:%d/%d person' % (i, len(X)), file=log_attack, flush=True)

    print("* Original: " + str(sample), file=log_attack, flush=True)

    print("  Original label: %d" % label, file=log_attack, flush=True)

    st = time.time()
    best_mf_process, best_greedy_set_process, \
    best_changed_set_process, best_iteration, robust_flag, orig_prob, best_greedy_set, best_greedy_set_visit_idx, \
    best_greedy_set_best_temp_funccall, best_prob, \
    best_object, best_changed_set, best_num_changed, best_flip_funccall, best_flip_set, best_flip_object, \
    best_failure_set = attacker.attack(sample, label, failure_set)
    print("Orig_Prob = " + str(orig_prob), file=log_attack, flush=True)
    if robust_flag == -1:
        print('Original Classification Error', file=log_attack, flush=True)
    else:
        print("* Result: ", file=log_attack, flush=True)
    et = time.time()
    all_t = et - st

    for j in range(4):
        if robust_flag == 1:
            print("This sample is robust.", file=log_attack, flush=True)
            break
        if robust_flag == -1:
            break
        st = time.time()

        mf_process, greedy_set_process, changed_set_process, iteration, \
        robust_flag, orig_prob, greedy_set, greedy_set_visit_idx, greedy_set_best_temp_funccall, prob, object, \
        changed_set, num_changed, flip_funccall, flip_set, flip_object, failure_set = attacker.attack(sample, label,
                                                                                                      failure_set)

        et = time.time()
        temp_t = et - st

        if iteration > best_iteration:
            best_mf_process = mf_process
            best_greedy_set_process = greedy_set_process
            best_changed_set_process = changed_set_process
            best_iteration = iteration
            best_greedy_set = greedy_set
            best_greedy_set_visit_idx = greedy_set_visit_idx
            best_greedy_set_best_temp_funccall = greedy_set_best_temp_funccall
            best_prob = prob
            best_object = object
            best_changed_set = changed_set
            best_num_changed = num_changed
            best_flip_funccall = flip_funccall
            best_flip_set = flip_set
            best_flip_object = flip_object
            best_failure_set = failure_set
            all_t = temp_t

    if robust_flag != -1:
        index += 1
        print('mf_process:', best_mf_process, file=log_attack, flush=True)
        print('greedy_set_process:', best_greedy_set_process, file=log_attack, flush=True)
        print('changed_set_process:', best_changed_set_process, file=log_attack, flush=True)
        print("  Number of iterations for this: " + str(best_iteration), file=log_attack, flush=True)
        print('greedy_set: ', file=log_attack, flush=True)
        print(best_greedy_set, file=log_attack, flush=True)
        print('greedy_set_visit_idx: ', file=log_attack, flush=True)
        print(best_greedy_set_visit_idx, file=log_attack, flush=True)
        print('greedy_funccall:', file=log_attack, flush=True)
        print(best_greedy_set_best_temp_funccall, file=log_attack, flush=True)
        print('best_prob = ' + str(best_prob), file=log_attack, flush=True)
        print('best_object = ' + str(best_object), file=log_attack, flush=True)
        print('changed set:', file=log_attack, flush=True)
        print(best_changed_set, file=log_attack, flush=True)
        print("  Number of changed codes: %d" % best_num_changed, file=log_attack, flush=True)
        print("failure_set:", file=log_attack, flush=True)
        print(best_failure_set, file=log_attack, flush=True)
        print(" Time: " + str(all_t), file=log_attack, flush=True)
        if robust_flag == 0:
            print('flip_funccall:', file=log_attack, flush=True)
            print(best_flip_funccall, file=log_attack, flush=True)
            print('flip_set:', file=log_attack, flush=True)
            print(best_flip_set, file=log_attack, flush=True)
            print('flip_object = ', best_flip_object, file=log_attack, flush=True)
            print(" The cardinality of S: " + str(len(best_greedy_set)), file=log_attack, flush=True)
        else:
            print(" The cardinality of S: " + str(len(best_greedy_set)) + ', but timeout', file=log_attack,
                  flush=True)

        mf_process_all.append(copy.deepcopy(best_mf_process))
        greedy_set_process_all.append(copy.deepcopy(best_greedy_set_process))
        changed_set_process_all.append(copy.deepcopy(best_changed_set_process))

        iterations_all.append(best_iteration)
        robust_flag_all.append(robust_flag)

        orignal_funccalls_all.append(copy.deepcopy(X[i].tolist()))
        orignal_labels_all.append(label)
        orignal_g_all.append(orig_prob)

        final_greedy_set_all.append(copy.deepcopy(best_greedy_set))
        final_greedy_set_visit_idx_all.append(copy.deepcopy(best_greedy_set_visit_idx))
        final_funccall_all.append(copy.deepcopy(best_greedy_set_best_temp_funccall))
        final_g_all.append(best_prob)
        final_mf_all.append(best_object)
        final_changed_set_all.append(copy.deepcopy(best_changed_set))
        final_changed_num_all.append(best_num_changed)

        if robust_flag == 0:
            flip_funccall_all.append(copy.deepcopy(best_flip_funccall))
            flip_set_all.append(copy.deepcopy(best_flip_set))
            flip_mf_all.append(best_flip_object)
            flip_sample_original_label_all.append(label)
            flip_sample_index_all.append(index)

        failue_set_all.append(copy.deepcopy(best_failure_set))

    pickle.dump(mf_process_all,
                open('../Logs/malware/%s/gradmin_mf_process.pickle' % Model_Type, 'wb'))
    pickle.dump(greedy_set_process_all,
                open('../Logs/malware/%s/gradmin_greedy_set_process.pickle' % Model_Type, 'wb'))
    pickle.dump(changed_set_process_all,
                open('../Logs/malware/%s/gradmin_changed_set_process.pickle' % Model_Type, 'wb'))
    pickle.dump(iterations_all,
                open('../Logs/malware/%s/gradmin_iteration.pickle' % Model_Type, 'wb'))
    pickle.dump(robust_flag_all,
                open('../Logs/malware/%s/gradmin_robust_flag.pickle' % Model_Type, 'wb'))
    pickle.dump(orignal_funccalls_all,
                open('../Logs/malware/%s/gradmin_original_funccall.pickle' % Model_Type, 'wb'))
    pickle.dump(orignal_labels_all,
                open('../Logs/malware/%s/gradmin_original_label.pickle' % Model_Type, 'wb'))
    pickle.dump(orignal_g_all,
                open('../Logs/malware/%s/gradmin_original_g.pickle' % Model_Type, 'wb'))
    pickle.dump(final_greedy_set_all,
                open('../Logs/malware/%s/gradmin_greedy_set.pickle' % Model_Type, 'wb'))
    pickle.dump(final_greedy_set_visit_idx_all,
                open('../Logs/malware/%s/gradmin_feature_greedy_set.pickle' % Model_Type, 'wb'))
    pickle.dump(final_funccall_all,
                open('../Logs/malware/%s/gradmin_modified_funccall.pickle' % Model_Type, 'wb'))
    pickle.dump(final_g_all,
                open('../Logs/malware/%s/gradmin_final_probs.pickle' % Model_Type, 'wb'))
    pickle.dump(final_mf_all,
                open('../Logs/malware/%s/gradmin_final_mf.pickle' % Model_Type, 'wb'))
    pickle.dump(final_changed_set_all,
                open('../Logs/malware/%s/gradmin_changed_set.pickle' % Model_Type, 'wb'))
    pickle.dump(final_changed_num_all,
                open('../Logs/malware/%s/gradmin_changed_num.pickle' % Model_Type, 'wb'))
    pickle.dump(flip_funccall_all,
                open('../Logs/malware/%s/gradmin_flip_funccall.pickle' % Model_Type, 'wb'))
    pickle.dump(flip_set_all,
                open('../Logs/malware/%s/gradmin_flip_set.pickle' % Model_Type, 'wb'))
    pickle.dump(flip_mf_all,
                open('../Logs/malware/%s/gradmin_flip_mf.pickle' % Model_Type, 'wb'))
    pickle.dump(flip_sample_original_label_all,
                open('../Logs/malware/%s/gradmin_flip_sample_original_label.pickle' % Model_Type, 'wb'))
    pickle.dump(flip_sample_index_all,
                open('../Logs/malware/%s/gradmin_flip_sample_index.pickle' % Model_Type, 'wb'))
    pickle.dump(failue_set_all,
                open('../Logs/malware/%s/gradmin_failure_set.pickle' % Model_Type, 'wb'))
