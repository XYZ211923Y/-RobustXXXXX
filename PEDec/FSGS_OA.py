from tools import *
from model import *
import random
import time
from copy import deepcopy
import argparse

parser = argparse.ArgumentParser(description='EHR with LSTM in PyTorch ')    #创建parser对象
parser.add_argument('--File_index', default=0, type=int, help='file index')
parser.add_argument('--QueryCap', default='FSGS_OA', type=str, help='query capabilities')
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

print(TITLE)
print(TITLE, file=log_f, flush=True)

attack_discreteData = load_data(args.data_path)
data = attack_discreteData[0]
label = attack_discreteData[1]
num_uniqFeature = len(data[0])
# load model
# print('Load the CNN model', file=log_f, flush=True)
if MODEL_TYPE == 'nonsub':
    net = Net_0D(num_uniqFeature).cuda()
    net.load_state_dict(torch.load('Output/nonsub_1e-06.30'))
    net.eval()
elif MODEL_TYPE == 'sub':
    net = Net_0D(num_uniqFeature).cuda()
    net.load_state_dict(torch.load('Output/net_weight_nopre_embed_sub/1e-06.450'))
    net.eval()
CEloss = nn.CrossEntropyLoss().cuda()

def Getpred(input=[], label=label):
    batch_attack_data= np.array([list(GetweightG(input,num_uniqFeature))])
    weight = torch.unsqueeze(torch.tensor(batch_attack_data), dim=1).cuda()

    logit = net(weight).cuda()
    pred_label = torch.max(logit[0].cpu().detach(), 0)[1].numpy()
    logit = logit[0][int(label)].cpu().detach().numpy()

    return logit,pred_label

def Getpred_M(input=[],label=1):
    weight = torch.unsqueeze(torch.tensor(input), dim=1).cuda()

    logit = net(weight).cuda()
    pred_value = logit.cpu().detach().numpy()[:,int(label)]

    return pred_value

def RSelectCode(topk_set_feature_index,Set_residual,SetGrad_min_index,Set_c,ori_label):
    SelectCode = Set_residual[topk_set_feature_index]
    Selectset = Set_c[int(SetGrad_min_index[topk_set_feature_index])]
    set_att = UnionEle(Selectset, [SelectCode])
    att_data = SetInsert(ori_data, set_att)
    logit,_ = Getpred(input=att_data, label=ori_label)

    return logit,SelectCode,Selectset,set_att

success_num = 0
NoAttack_num = 0
F = []
g = []
F_V = []
Total_iteration = 0
Total_targCode = 0

i = 0
for ori_data, ori_label in zip(data, label):
    ori_data = list(np.where(ori_data > 0.5)[0])
    time_start = time.time()
    i = i + 1
    robust_flag = 1 #(1 is robust, 0 is Noneed attack, -1 is not robust)
    print('-------------the number %d/%d----------------' % (i,len(label)), file=log_f, flush=True)
    pred_value_ori, pred_label_ori = Getpred(input=ori_data, label=ori_label)
    if pred_label_ori != ori_label:
        NoAttack_num += 1
        print('ori_classifier predicts wrong!', file=log_f, flush=True)
        robust_tag = 0
        continue

    allCode = list(range(num_uniqFeature))
    g_target = []  # The (:the best value of target F function).
    F_S = []  # The (S:the best value of target F function).  it is different from the F_u(S) = F(S+u) - F(S)
    F_value = []
    Set_candidate = []  # the candidate set is S after selecting code process
    Set_target = [()]  # the target set is best chosen attck set_u_att under S
    Set_delet = []

    Set_c = [()]
    iteration = 0
    Set_residual = DiffElem(allCode,Set_candidate)
    ScoreMin = pred_value_ori
    F_best = 100*(2*pred_value_ori -1)
    while robust_flag == 1:
        iteration +=1
        # print(("----the %dth---")%(iteration), file=log_f, flush=True)
        g_set = []  # this is the list of the value of g(set_u_att).
        code_cand = [] # this is the list of selected code of each set_
        set_u_att = []  # the attack set after selected feature  u under the set_
        F_set = []  # this is the list of the value of F_u(set_u_att). This is to prepare random select
        F_S_new = [] # this is the list of the value of F(set_u_att)

        batch_size = 4
        batch_num = len(Set_residual) // batch_size
        SetPred_min = 100*np.ones([batch_num * batch_size])
        SetPred_min_index = np.zeros([batch_num * batch_size])
        LOSS = []
        PRED = []
        GRADSET = []
        for set_ in Set_c:
            set_data = SetInsert(ori_data,set_)
            attack_matrix = GetAllAttck(set_data,Set_residual,num_uniqFeature)
            Set_pred = np.zeros([batch_num * batch_size])
            for index in range(batch_num):
                batch_attack_data = attack_matrix[batch_size * index: batch_size * (index + 1)]
                batch_attack_pred = Getpred_M(input=batch_attack_data)
                Set_pred[batch_size * index: batch_size * (index + 1)] = batch_attack_pred

            SetPred_min_index = SetPred_min_index + np.argmin([SetPred_min, Set_pred],axis=0) # the min set index (-1) set compared with former set
            SetPred_min = np.min([SetPred_min,Set_pred],axis = 0)   # the min grad value for set compared with former set

        pred_sort = np.argsort(SetPred_min)
        SetPred_min_index = SetPred_min_index - 1

        topk_set_feature_index = random.choice(pred_sort[-TopK:])

        pred_value_att, SelectCode, Selectset, set_att = RSelectCode(topk_set_feature_index,Set_residual, SetPred_min_index, Set_c,
                                                                     ori_label)

        if ScoreMin >= pred_value_att:
            ScoreMin = pred_value_att
            set_att = set_att
        else:
            set_att = Set_target[-1]


        F_new = ScoreMin
        F_set = F_new - F_best
        F_best = F_new

        Set_candidate.append(SelectCode)
        Set_target.append(set_att)

        F_S.append(deepcopy(Set_candidate))
        g_target.append(set_att)
        F_value.append(ScoreMin)

        Set_residual = DiffElem(DiffElem(allCode,Set_candidate),Set_delet)
        Set_c = list(powerset(Set_candidate))

        if ScoreMin <= TAU:
            success_num += 1
            print('Attack Success', file=log_f, flush=True)
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

    print("Target Score",ScoreMin, file=log_f, flush=True)

    print("  Number of searched codes: %d" % (len(Set_candidate)), file=log_f, flush=True)
    print("  Number of changed codes: %d" % (len(set_att)), file=log_f, flush=True)

    print("  Number of iterations for this: " + str(iteration), file=log_f, flush=True)

    print(" Time: " + str(time.time() - time_start), file=log_f, flush=True)

    print("* SUCCESS Number NOW: %d " % (success_num), file=log_f, flush=True)
    print("* NoAttack Number NOW: %d " % (NoAttack_num), file=log_f, flush=True)

    pickle.dump(F,
                open('./AttackCodes/%s/%s_k=%d_t=%s_s=%d_F.pickle' % (MODEL_TYPE, Algo_TYPE, TopK, str(TAU), SECONDS),
                     'wb'))
    pickle.dump(g,
                open('./AttackCodes/%s/%s_k=%d_t=%s_s=%d_g.pickle' % (MODEL_TYPE, Algo_TYPE, TopK, str(TAU), SECONDS),
                     'wb'))
    pickle.dump(F_V,
                open('./AttackCodes/%s/%s_k=%d_t=%s_s=%d_F_V.pickle' % (MODEL_TYPE, Algo_TYPE, TopK, str(TAU), SECONDS),
                     'wb'))

    print("--- Total Success Number: " + str(success_num) + " ---", file=log_f, flush=True)
    print("--- Total No Attack Number: " + str(NoAttack_num) + " ---", file=log_f, flush=True)
    if (len(label) - NoAttack_num) != 0:
        print("--- success Ratio: " + str(success_num / (len(label) - NoAttack_num)) + " ---", file=log_f, flush=True)
        print("--- Mean Iteration: " + str(Total_iteration / (len(label) - NoAttack_num)) + " ---", file=log_f,
              flush=True)
        print("--- Mean TargetCode: " + str(Total_targCode / (len(label) - NoAttack_num)) + " ---", file=log_f,
              flush=True)


print(TITLE)
print(TITLE, file=log_f, flush=True)


