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

    def eval_object(self, eval_funccall, current_object, greedy_set, orig_label):
        best_temp_funccall = copy.deepcopy(eval_funccall)
        candidate_lists = []
        success_flag = 1
        funccall_lists = []
        sets_lists = []
        changed_set = set()
        change_flag = 0
        worst_object = current_object
        label_set = {0, 1, 2}
        label_set.remove(orig_label)
        list_label_set = list(label_set)

        eval_pred, eval_g, eval_h, eval_h1, eval_h2 = self.classify(eval_funccall, orig_label)
        object = eval_h - eval_g
        if object > 0:
            return object, eval_funccall, 0, changed_set
        if object >= worst_object:
            change_flag = 1
            worst_object = object
        # candidate_lists contains all the non-empty subsets of greedy_set
        if greedy_set:
            for i in range(1, len(greedy_set) + 1):
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

        batch_size = 20
        n_batches = int(np.ceil(float(len(funccall_lists)) / float(batch_size)))
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

            logit = self.rnn(model_input, torch.tensor(t_diagnosis_codes).cuda())
            logit = logit.data.cpu().numpy()
            subsets_g = logit[:, orig_label]
            subsets_h1 = logit[:, list_label_set[0]]
            subsets_h2 = logit[:, list_label_set[1]]
            subsets_h = np.max([subsets_h1, subsets_h2], axis=0)
            subsets_object = subsets_h - subsets_g
            max_object = np.max(subsets_object)
            max_index = np.argmax(subsets_object)

            if max_object >= worst_object:
                change_flag = 1
                worst_object = max_object
                best_temp_funccall = copy.deepcopy(funccall_lists[batch_size * index + max_index])
                changed_set = copy.deepcopy(sets_lists[batch_size * index + max_index])

        if change_flag == 0:
            success_flag = 2

        if worst_object > 0:
            success_flag = 0
            # print(worst_object)

        return worst_object, best_temp_funccall, success_flag, changed_set

    def attack(self, funccall, y):
        print()
        st = time.time()
        success_flag = 1

        orig_pred, orig_g, orig_h, orig_h1, orig_h2 = self.classify(funccall, y)

        greedy_set = set()
        greedy_set_visit_idx = set()
        greedy_set_best_temp_funccall = funccall
        final_changed_set = set()
        flip_set = set()

        mf_process = []
        greedy_set_process = []
        changed_set_process = []

        mf_process.append(np.float(orig_h - orig_g))

        n_changed = 0
        iteration = 0
        robust_flag = 0

        current_object = orig_h - orig_g
        flip_object = 0
        flip_funccall = funccall
        max_pos = (0, 0)
        max_changed_set = set()
        max_funccall = funccall
        if current_object > 0:
            robust_flag = -1
            print("Original classification error")

            return mf_process, greedy_set_process, changed_set_process, \
                   iteration, robust_flag, np.float(orig_g), greedy_set, greedy_set_visit_idx, \
                   greedy_set_best_temp_funccall.tolist(), np.float(orig_g), np.float(current_object), \
                   final_changed_set, n_changed, flip_funccall.tolist(), flip_set, np.float(flip_object)

        print(current_object)
        while success_flag == 1:
            iteration += 1
            success_flag = 1
            candidate_objects = []
            candidate_funccalls = []
            candidate_poses = []
            candidate_changed_sets = []

            for visit_idx in range(len(funccall)):
                if visit_idx in greedy_set_visit_idx:
                    continue
                worst_object_cate = -2
                best_pos_cate = -1
                best_temp_funccall_cate = funccall
                best_temp_changed_set_cate = set()
                for code_idx in range(self.n_diagnosis_codes):

                    pos = (visit_idx, code_idx)
                    if pos in greedy_set:
                        continue
                    if code_idx == funccall[visit_idx]:
                        continue

                    eval_funccall = copy.deepcopy(funccall)
                    eval_funccall[visit_idx] = code_idx
                    worst_object, temp_funccall, success_flag_temp, temp_changed_set = self.eval_object(eval_funccall,
                                                                                                        current_object,
                                                                                                        greedy_set, y)
                    if success_flag_temp == 1:
                        temp_changed_set.add(pos)

                    elif success_flag_temp == 2:
                        temp_funccall = greedy_set_best_temp_funccall
                        temp_changed_set = final_changed_set

                    else:
                        success_flag = 0

                    if worst_object > worst_object_cate:
                        worst_object_cate = worst_object
                        best_pos_cate = pos
                        best_temp_funccall_cate = temp_funccall
                        best_temp_changed_set_cate = temp_changed_set

                candidate_objects.append(worst_object_cate)
                candidate_funccalls.append(best_temp_funccall_cate)
                candidate_poses.append(best_pos_cate)
                candidate_changed_sets.append(best_temp_changed_set_cate)

            index = np.argmax(candidate_objects)
            max_object = np.max(candidate_objects)
            max_pos = candidate_poses[index]
            max_funccall = candidate_funccalls[index]
            max_changed_set = candidate_changed_sets[index]
            print(iteration)
            print(max_object)

            sorted_index = np.argsort(candidate_objects)[::-1]
            idx = 0
            if iteration == 1 and success_flag == 0:
                for i in range(len(candidate_objects)):
                    if candidate_objects[sorted_index[i]] >= 0:
                        continue
                    idx = i
                    break
                greedy_set_best_temp_funccall = candidate_funccalls[idx]
                final_changed_set = {candidate_poses[idx]}
                changed_set_process.append(copy.deepcopy(final_changed_set))

            if success_flag == 1:
                greedy_set.add(max_pos)
                greedy_set_visit_idx.add(max_pos[0])
                if max_object > current_object:
                    greedy_set_best_temp_funccall = max_funccall
                    current_object = max_object
                    final_changed_set = max_changed_set
                changed_set_process.append(copy.deepcopy(final_changed_set))
                pred, g, h, h1, h2 = self.classify(greedy_set_best_temp_funccall, y)
                mf_process.append(np.float(h - g))
                greedy_set_process.append(copy.deepcopy(greedy_set))
                print(greedy_set)

            if (time.time() - st) > 3600:
                success_flag = -1
                robust_flag = 1

        for i in range(len(greedy_set_best_temp_funccall)):
            if greedy_set_best_temp_funccall[i] != funccall[i]:
                n_changed += 1

        pred, g, h, h1, h2 = self.classify(greedy_set_best_temp_funccall, y)
        if robust_flag == 0:
            flip_set = copy.copy(max_changed_set)
            flip_set.add(max_pos)
            flip_funccall = copy.deepcopy(max_funccall)
            flip_pred, flip_g, flip_h, flip_h1, flip_h2 = self.classify(flip_funccall, y)
            flip_object = flip_h - flip_g
            mf_process.append(np.float(flip_object))

        print("Modified_set:", flip_set)
        print(flip_funccall)
        print()

        return mf_process, greedy_set_process, changed_set_process, \
               iteration, robust_flag, np.float(orig_g), greedy_set, greedy_set_visit_idx, \
               greedy_set_best_temp_funccall.tolist(), np.float(g), np.float(current_object), final_changed_set, \
               n_changed, flip_funccall.tolist(), flip_set, np.float(flip_object)


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

log_attack = open(
    '../Logs/malware/%s/greedmax_Attack.bak' % Model_Type, 'w+')
attacker = Attacker(best_parameters_file, log_attack, emb_weights)
index = -1
for i in range(len(X)):
    print(i)
    print("---------------------- %d --------------------" % i, file=log_attack, flush=True)

    sample = X[i]

    label, _, _, _, _ = attacker.classify(sample, 0)
    label = np.int(label)

    print('* Processing:%d/%d person' % (i, len(X)), file=log_attack, flush=True)

    print("* Original: " + str(sample), file=log_attack, flush=True)

    print("  Original label: %d" % label, file=log_attack, flush=True)

    st = time.time()
    best_mf_process, best_greedy_set_process, best_changed_set_process, best_iteration, robust_flag, orig_prob, \
    best_greedy_set, best_greedy_set_visit_idx, best_greedy_set_best_temp_funccall, best_prob, best_object, \
    best_changed_set, best_num_changed, best_flip_funccall, best_flip_set, best_flip_object, = attacker.attack(sample, label)
    print("Orig_Prob = " + str(orig_prob), file=log_attack, flush=True)
    if robust_flag == -1:
        print('Original Classification Error', file=log_attack, flush=True)
    else:
        print("* Result: ", file=log_attack, flush=True)
    et = time.time()
    all_t = et - st

    if robust_flag == 1:
        print("This sample is robust.", file=log_attack, flush=True)

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

    pickle.dump(mf_process_all,
                open('../Logs/malware/%s/greedmax_mf_process.pickle' % Model_Type, 'wb'))
    pickle.dump(greedy_set_process_all,
                open('../Logs/malware/%s/greedmax_greedy_set_process.pickle' % Model_Type, 'wb'))
    pickle.dump(changed_set_process_all,
                open('../Logs/malware/%s/greedmax_changed_set_process.pickle' % Model_Type, 'wb'))
    pickle.dump(iterations_all,
                open('../Logs/malware/%s/greedmax_iteration.pickle' % Model_Type, 'wb'))
    pickle.dump(robust_flag_all,
                open('../Logs/malware/%s/greedmax_robust_flag.pickle' % Model_Type, 'wb'))
    pickle.dump(orignal_funccalls_all,
                open('../Logs/malware/%s/greedmax_original_funccall.pickle' % Model_Type, 'wb'))
    pickle.dump(orignal_labels_all,
                open('../Logs/malware/%s/greedmax_original_label.pickle' % Model_Type, 'wb'))
    pickle.dump(orignal_g_all,
                open('../Logs/malware/%s/greedmax_original_g.pickle' % Model_Type, 'wb'))
    pickle.dump(final_greedy_set_all,
                open('../Logs/malware/%s/greedmax_greedy_set.pickle' % Model_Type, 'wb'))
    pickle.dump(final_greedy_set_visit_idx_all,
                open('../Logs/malware/%s/greedmax_feature_greedy_set.pickle' % Model_Type, 'wb'))
    pickle.dump(final_funccall_all,
                open('../Logs/malware/%s/greedmax_modified_funccall.pickle' % Model_Type, 'wb'))
    pickle.dump(final_g_all,
                open('../Logs/malware/%s/greedmax_final_probs.pickle' % Model_Type, 'wb'))
    pickle.dump(final_mf_all,
                open('../Logs/malware/%s/greedmax_final_mf.pickle' % Model_Type, 'wb'))
    pickle.dump(final_changed_set_all,
                open('../Logs/malware/%s/greedmax_changed_set.pickle' % Model_Type, 'wb'))
    pickle.dump(final_changed_num_all,
                open('../Logs/malware/%s/greedmax_changed_num.pickle' % Model_Type, 'wb'))
    pickle.dump(flip_funccall_all,
                open('../Logs/malware/%s/greedmax_flip_funccall.pickle' % Model_Type, 'wb'))
    pickle.dump(flip_set_all,
                open('../Logs/malware/%s/greedmax_flip_set.pickle' % Model_Type, 'wb'))
    pickle.dump(flip_mf_all,
                open('../Logs/malware/%s/greedmax_flip_mf.pickle' % Model_Type, 'wb'))
    pickle.dump(flip_sample_original_label_all,
                open('../Logs/malware/%s/greedmax_flip_sample_original_label.pickle' % Model_Type, 'wb'))
    pickle.dump(flip_sample_index_all,
                open('../Logs/malware/%s/greedmax_flip_sample_index.pickle' % Model_Type, 'wb'))

