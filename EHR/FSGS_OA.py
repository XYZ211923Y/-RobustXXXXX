import time
import torch.nn as nn
import rnn_tools
import rnn_model
from torch.autograd import Variable
from tools import *
import random
import argparse

parser = argparse.ArgumentParser(description='FSGS_KA')    #创建parser对象
parser.add_argument('--File_index', default=0, type=int, help='file index')
parser.add_argument('--QueryCap', default='FSGS_KA', type=str, help='query capabilities')
parser.add_argument('--threshold', type=int, default=0.5, help='threshold')
parser.add_argument('--TopK', type=int, default=10, help='TopK')
parser.add_argument('--time_limit', type=int, default=3600, help='time_limit')
parser.add_argument('--model', type=str, default='nonsub', help='model')
parser.add_argument('--data_path', default='dataset/5000_attackdata_0.2.pickle',type=str, help='attack_file')
parser.add_argument('--seed', type=int, default=666, help='random seed (default: 100)')

args=parser.parse_args()#解析参数，此处args是一个命名空间列表
print(args)

args=parser.parse_args()#解析参数，此处args是一个命名空间列表
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)

QC_TYPE = args.QueryCap
TAU = args.threshold
TopK = args.TopK
SECONDS= args.time_limit
File_index = args.File_index
MODEL_TYPE = args.model

Algo_TYPE = ' '+ QC_TYPE +'_'+str(File_index)
log_f = open('./Logs/%s/%s_k=%d_t=%s_s=%d.bak'% (MODEL_TYPE,Algo_TYPE, TopK, str(TAU), SECONDS), 'w+')
TITLE = '=== ' + MODEL_TYPE + Algo_TYPE + ' target prob = ' + str(TAU) + ' k = ' \
        + str(TopK) + ' time = ' + str(SECONDS) + ' ==='



class Attacker(object):
    def __init__(self, options, emb_weights):
        print("Loading pre-trained classifier...", file=log_f, flush=True)

        self.model = rnn_model.LSTM(options, emb_weights).cuda()
        self.model2 = rnn_model.LSTM(options, emb_weights).cuda()

        if MODEL_TYPE == 'sub':
            self.model.load_state_dict(torch.load('./Classifiers/Submodular_lstm.49'))  # abs
            self.model2.load_state_dict(torch.load('./Classifiers/Submodular_lstm.49'))
        elif MODEL_TYPE == 'nonsub':
            self.model.load_state_dict(torch.load('./Classifiers/Nonsubmodular_lstm.42'))  # positive and negative
            self.model2.load_state_dict(torch.load('./Classifiers/Nonsubmodular_lstm.42'))
        elif MODEL_TYPE == 'weaksub':
            self.model.load_state_dict(torch.load('./Classifiers/Weaksubmodular_lstm.43'))  # weak abs
            self.model2.load_state_dict(torch.load('./Classifiers/weaksubmodular_lstm.43'))

        self.model.eval()
        self.model2.eval()

        self.criterion = nn.CrossEntropyLoss()

    def classify(self, person,y):

        model_input, weight_of_embed_codes = self.input_handle(person)

        logit = self.model2(model_input, weight_of_embed_codes)

        pred = torch.max(logit[0].cpu().detach(), 0)[1].numpy()
        prob = logit[0][int(y)].cpu().detach().numpy()

        return pred, prob

    def classifyM(self, person,y):

        model_input, weight_of_embed_codes = self.input_handleM(person)

        logit = self.model2(model_input, weight_of_embed_codes)

        prob = logit.cpu().detach().numpy()[:,int(y)]

        return prob

    def input_handle(self, person):
        t_diagnosis_codes = rnn_tools.pad_matrix(person)
        model_input = deepcopy(t_diagnosis_codes)
        for i in range(len(model_input)):
            for j in range(len(model_input[i])):
                idx = 0
                for k in range(len(model_input[i][j])):
                    model_input[i][j][k] = idx
                    idx += 1

        model_input = Variable(torch.LongTensor(model_input))
        return model_input.transpose(0, 1).cuda(), torch.tensor(t_diagnosis_codes).transpose(0, 1).cuda()

    def input_handleM(self, person):
        t_diagnosis_codes,t_labels, batch_mask = rnn_tools.pad_matrix_M(person,[])
        model_input = copy.copy(t_diagnosis_codes)
        for i in range(len(model_input)):
            for j in range(len(model_input[i])):
                idx = 0
                for k in range(len(model_input[i][j])):
                    model_input[i][j][k] = idx
                    idx += 1

        model_input = Variable(torch.LongTensor(model_input).cuda())
        t_diagnosis_codes = torch.tensor(t_diagnosis_codes).cuda()

        return model_input, t_diagnosis_codes


    def forward_lstm(self, weighted_embed_codes, model):
        x = model.relu(weighted_embed_codes)
        x = torch.mean(x, dim=2)
        h0 = Variable(torch.FloatTensor(torch.randn(1, x.size()[1], x.size()[2])))
        c0 = Variable(torch.FloatTensor(torch.randn(1, x.size()[1], x.size()[2])))
        output, h_n = model.lstm(x, (h0, c0))
        embedding, attn_weights = model.attention(output.transpose(0, 1))
        x = model.dropout(embedding)  # (n_samples, hidden_size)

        logit = model.fc(x)  # (n_samples, n_labels)

        logit = model.softmax(logit)
        return logit


    def getMatrix(self,set_data,visit_R,code_R):
        attack_matrix = []
        for visit, code in zip(visit_R,code_R):
            data = deepcopy(set_data)
            if str(visit) != str(()):
                data[visit] = SetInsert(set_data[visit], [code])
            attack_matrix.append(data)
        return attack_matrix

    def RSelectCode(self,topk_set_feature_index,Set_residual, SetGrad_min_index, Set_c,SetPred_min_R):
        SelectCode = Set_residual[topk_set_feature_index]
        prob_att = SetPred_min_R[topk_set_feature_index]

        SetGrad_min_index = SetGrad_min_index - 1
        Selectset = Set_c[int(SetGrad_min_index[topk_set_feature_index])]

        set_att = UnionEle(Selectset, [SelectCode])

        return prob_att, SelectCode, Selectset, set_att

    def attack(self, person, y,Set_residual,Set_c):

        batch_size = 5
        batch_num = len(Set_residual) // batch_size
        # batch_num = 2

        SetPred_min_R = (100) * np.ones([batch_num * batch_size])
        SetPred_min_R_index = np.zeros([batch_num * batch_size],dtype='int')
        # Set_c_visit = [(1,2), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
        # Set_c_code = [(4,5), (5,), (6,), (4, 5), (4, 6), (5, 6), (4, 5, 6)]

        for set_ in Set_c:
            set_data = SetInsert_vect(person,set_)

            visit_R, code_R = FeatureTo2D(Set_residual)
            attack_matrix = self.getMatrix(set_data,visit_R,code_R)

            Set_pred_R = np.zeros([batch_num * batch_size])
            for index in range(batch_num):
                batch_attack_data = attack_matrix[batch_size * index: batch_size * (index + 1)]
                batch_attack_prob = self.classifyM(batch_attack_data, y)
                Set_pred_R[batch_size * index: batch_size * (index + 1)] = batch_attack_prob


            SetPred_min_R_index = SetPred_min_R_index + np.argmin([SetPred_min_R,Set_pred_R],
                                                                  axis=0)  # the min set index (-1) set compared with former set
            SetPred_min_R = np.min([SetPred_min_R,Set_pred_R],
                                     axis=0)  # the min grad value for set compared with former set

        topk_set_feature_index = np.argsort(SetPred_min_R)[-TopK:]
        feature_index = random.choice(topk_set_feature_index)

        prob_att, SelectFeat, Selectset, set_att = self.RSelectCode(feature_index,Set_residual, SetPred_min_R_index, Set_c,SetPred_min_R)

        return SelectFeat,Selectset,prob_att,set_att


def main(emb_weights, training_file, validation_file,
         testing_file, n_diagnosis_codes, n_labels,
         batch_size, dropout_rate,
         L2_reg, n_epoch, log_eps, n_claims, visit_size, hidden_size,
         use_gpu, model_name):
    options = locals().copy()
    print("Loading dataset...", file=log_f, flush=True)
    test = rnn_tools.load_data(training_file, validation_file, testing_file)

    n_people = len(test[0])

    attacker = Attacker(options, emb_weights)

    n_success = 0
    n_fail = 0

    total_node_change = 0

    n_iteration = 0

    saving_time = {}

    attack_code_dict = {}

    NoAttack_num = 0
    success_num = 0
    success_data = []
    danger_data = []
    sample_index =[]
    success_label = []

    F = []
    g = []
    F_V = []
    Total_iteration = 0
    Total_targCode = 0

    for i in range(File_index * 100, (File_index + 1) * 100):
        print("-------- %d ---------" % (i), file=log_f, flush=True)

        person = test[0][i]

        y = test[1][i]

        n_visit = len(person)

        print('* Processing:%d/%d person, number of visit for this person: %d' % (i, n_people, n_visit), file=log_f,
              flush=True)

        print("* Original: " + str(person), file=log_f, flush=True)

        print("  Original label: %d" % (y), file=log_f, flush=True)

        time_start = time.time()
        # changed_person, score, num_changed, success_flag, iteration, changed_pos = attacker.attack(person, y)
        robust_flag = 1
        orig_pred, orig_prob = attacker.classify(person, y)
        if orig_pred != y:
            NoAttack_num += 1
            print('ori_classifier predicts wrong!', file=log_f, flush=True)
            robust_tag = 0
            continue

        Set_candidate = []
        Set_delet = []

        Set_target = [()]
        Set_c = [()]

        ScoreMin = orig_prob

        F_S =[]
        g_target = []
        F_value = []

        iteration = 0
        allCode = list(range(len(person)*4130))
        Set_residual = DiffElem(allCode, Set_candidate)

        while robust_flag == 1:
            iteration +=1
            SelectFeat, Selectset,prob_att,set_att = attacker.attack(person, y,Set_residual,Set_c)

            if prob_att <= ScoreMin:
                ScoreMin = prob_att
            else:
                set_att = Set_target[-1]

            Set_candidate.append(SelectFeat)
            Set_target.append(set_att)

            F_S.append(deepcopy(Set_candidate))
            g_target.append(set_att)
            F_value.append(ScoreMin)

            Set_residual = DiffElem(DiffElem(allCode, Set_candidate), Set_delet)
            Set_c = list(powerset(Set_candidate))

            if ScoreMin < TAU:
                success_num += 1
                sample_index.append(i)
                success_data.append(SetInsert_vect(person,Set_target[-1]))
                danger_data.append(SetInsert_vect(person,Set_target[-2]))
                success_label.append(y)
                print('Attack Success', file=log_f, flush=True)
                break
            if len(Set_residual) == 0:
                print("Searched all the features", file=log_f, flush=True)
                break
            time_end = time.time()
            time_Dur = time_end - time_start
            if time_Dur > SECONDS:
                print('The time is over', file=log_f, flush=True)
                break


        F.append(deepcopy(F_S))
        g.append(deepcopy(g_target))
        F_V.append(deepcopy(F_value))

        Total_iteration += len(Set_candidate)
        Total_targCode += len(set_att)
        print("* Result: ", file=log_f, flush=True)
        print('Searched Features',F_S, file=log_f, flush=True)
        print('Target Features',g_target, file=log_f, flush=True)
        print('Searched Prediction Score', F_value, file=log_f, flush=True)
        print("Target Score", ScoreMin, file=log_f, flush=True)

        print("  Number of searched features: %d" % (len(Set_candidate)), file=log_f, flush=True)
        print("  Number of changed features: %d" % (len(set_att)), file=log_f, flush=True)

        print("  Number of iterations for this: " + str(iteration), file=log_f, flush=True)

        print(" Time: " + str(time.time() - time_start), file=log_f, flush=True)

        print("* SUCCESS Number NOW: %d " % (success_num), file=log_f, flush=True)
        print("* NoAttack Number NOW: %d " % (NoAttack_num), file=log_f, flush=True)
        pickle.dump(F, open(
            './AttackCodes/%s/%s_k=%d_t=%s_s=%d_F.pickle' % (MODEL_TYPE, Algo_TYPE, TopK, str(TAU), SECONDS), 'wb'))
        pickle.dump(g, open(
            './AttackCodes/%s/%s_k=%d_t=%s_s=%d_g.pickle' % (MODEL_TYPE, Algo_TYPE, TopK, str(TAU), SECONDS), 'wb'))
        pickle.dump(F_V, open(
            './AttackCodes/%s/%s_k=%d_t=%s_s=%d_F_V.pickle' % (MODEL_TYPE, Algo_TYPE, TopK, str(TAU), SECONDS), 'wb'))
        pickle.dump(success_data, open(
            './AttackCodes/%s/%s_k=%d_t=%s_s=%d_success_data.pickle' % (MODEL_TYPE, Algo_TYPE, TopK, str(TAU), SECONDS),
            'wb'))
        pickle.dump(danger_data, open(
            './AttackCodes/%s/%s_k=%d_t=%s_s=%d_danger_data.pickle' % (MODEL_TYPE, Algo_TYPE, TopK, str(TAU), SECONDS),
            'wb'))
        pickle.dump(sample_index, open(
            './AttackCodes/%s/%s_k=%d_t=%s_s=%d_sample_index.pickle' % (MODEL_TYPE, Algo_TYPE, TopK, str(TAU), SECONDS),
            'wb'))
        pickle.dump(success_label, open('./AttackCodes/%s/%s_k=%d_t=%s_s=%d_success_label.pickle' % (
        MODEL_TYPE, Algo_TYPE, TopK, str(TAU), SECONDS), 'wb'))

        print("--- Total Success Number: " + str(success_num) + " ---", file=log_f, flush=True)
        print("--- Total No Attack Number: " + str(NoAttack_num) + " ---", file=log_f, flush=True)
        if (len(test[1]) - NoAttack_num) != 0:
            print("--- success Ratio: " + str(success_num / (len(test[1]) - NoAttack_num)) + " ---", file=log_f,
                  flush=True)
            print("--- Mean Iteration: " + str(Total_iteration / (len(test[1]) - NoAttack_num)) + " ---", file=log_f,
                  flush=True)
            print("--- Mean TargetCode: " + str(Total_targCode / (len(test[1]) - NoAttack_num)) + " ---", file=log_f,
                  flush=True)

    print(TITLE)
    print(TITLE, file=log_f, flush=True)


if __name__ == '__main__':
    print(TITLE, file=log_f, flush=True)
    print(TITLE)
    # parameters
    batch_size = 5
    dropout_rate = 0.5
    L2_reg = 0.001  # 0.001
    log_eps = 1e-8
    n_epoch = 50
    n_labels = 2  # binary classification
    visit_size = 70
    hidden_size = 70
    n_diagnosis_codes = 4130
    n_claims = 504

    use_gpu = False
    model_name = 'lstm'

    trianing_file = ''
    validation_file = ''
    testing_file = './Datasource/attack_data.pickle'

    emb_weights_char = torch.load("./SourceData/PretrainedEmbedding.4")['char_embeddings.weight']
    emb_weights_word = torch.load("./SourceData/PretrainedEmbedding.4")['word_embeddings.weight']

    ##################

    map_char_idx = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '.': 10, 'E': 11,
                    'V': 12, 'VAC': 13}

    tree = pickle.load(open('./SourceData/hf_dataset_270_code_dict.pickle', 'rb'))
    map_codeidx_charidx = {}

    for k in tree.keys():
        codeidx = tree[k]
        charidx = []

        code = str(k)
        len_code = len(code)

        if len_code == 7:
            for c in code:
                charidx.append(map_char_idx[c])

        elif len_code == 6:

            if code[0] == 'V':
                charidx.append(map_char_idx[code[0]])
                charidx.append(map_char_idx['VAC'])
                for i in range(1, 6):
                    charidx.append(map_char_idx[code[i]])

            elif code[0] == 'E':
                charidx.append(map_char_idx[code[0]])
                for i in range(1, 6):
                    charidx.append(map_char_idx[code[i]])
                charidx.append(map_char_idx['VAC'])

            else:
                charidx.append(map_char_idx['VAC'])
                for i in range(6):
                    charidx.append(map_char_idx[code[i]])

        elif len_code == 5:

            if code[0] == 'V':
                charidx.append(map_char_idx[code[0]])
                charidx.append(map_char_idx['VAC'])
                for i in range(1, 5):
                    charidx.append(map_char_idx[code[i]])
                charidx.append(map_char_idx['VAC'])

            else:
                charidx.append(map_char_idx['VAC'])
                for i in range(5):
                    charidx.append(map_char_idx[code[i]])
                charidx.append(map_char_idx['VAC'])

        elif len_code == 4:
            for i in range(4):
                charidx.append(map_char_idx[code[i]])
            charidx.append(map_char_idx['VAC'])
            charidx.append(map_char_idx['VAC'])
            charidx.append(map_char_idx['VAC'])

        elif len_code == 3:
            if code[0] == 'V':
                charidx.append(map_char_idx[code[0]])
                charidx.append(map_char_idx['VAC'])
                charidx.append(map_char_idx[code[1]])
                charidx.append(map_char_idx[code[2]])
                charidx.append(map_char_idx['VAC'])
                charidx.append(map_char_idx['VAC'])
                charidx.append(map_char_idx['VAC'])
            else:
                charidx.append(map_char_idx['VAC'])
                charidx.append(map_char_idx[code[0]])
                charidx.append(map_char_idx[code[1]])
                charidx.append(map_char_idx[code[2]])
                charidx.append(map_char_idx['VAC'])
                charidx.append(map_char_idx['VAC'])
                charidx.append(map_char_idx['VAC'])

        map_codeidx_charidx[codeidx] = charidx

    codes_embedding = []

    for i in range(4130):
        chars = map_codeidx_charidx[i]

        char_code_embedding = []
        for c in chars:
            c_embedding = emb_weights_char[c].tolist()
            char_code_embedding.append(c_embedding)

        char_code_embedding = np.reshape(char_code_embedding, (-1))

        word_embedding = np.array(emb_weights_word[i])

        code_embedding = 0.5 * char_code_embedding + 0.5 * word_embedding

        codes_embedding.append(code_embedding)
    ##################

    emb_weights = torch.tensor(codes_embedding, dtype=torch.float)
    main(emb_weights, trianing_file, validation_file,
         testing_file, n_diagnosis_codes, n_labels,
         batch_size, dropout_rate,
         L2_reg, n_epoch, log_eps, n_claims, visit_size, hidden_size,
         use_gpu, model_name)
