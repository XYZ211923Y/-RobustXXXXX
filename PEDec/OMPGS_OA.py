from tools import *
from model import *
import random
import time
from copy import deepcopy
import argparse

parser = argparse.ArgumentParser(description='OMPGS_OA')    #创建parser对象
parser.add_argument('--File_index', default=0, type=int, help='file index')
parser.add_argument('--QueryCap', default='OMPGS_OA', type=str, help='query capabilities')
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
    net.load_state_dict(torch.load('./Output/net_weight_nopre_embed/1e-06.30'))
    net.eval()
    net_grad = Net_0D(num_uniqFeature).cuda()
    net_grad.load_state_dict(torch.load('./Output/net_weight_nopre_embed/1e-06.30'))
    net_grad.eval()
elif MODEL_TYPE == 'sub':
    net = Net_0D(num_uniqFeature).cuda()
    net.load_state_dict(torch.load('Output/net_weight_nopre_embed_sub/1e-06.450'))
    net.eval()
    net_grad = Net_0D(num_uniqFeature).cuda()
    net_grad.load_state_dict(torch.load('Output/net_weight_nopre_embed_sub/1e-06.450'))
    net_grad.eval()

CEloss = nn.CrossEntropyLoss().cuda()

def SetInsert(ori_data,set_):
    union_ = UnionEle(ori_data, set_)
    same_ = SameElem(ori_data, set_)
    # get the attack data from the set_: # delete the same code and add the different code (ori_data and selected set_new)
    set_data_ = DiffElem(union_, same_)

    return set_data_

def GetGradient(input=[], label=label):
    # net_grad=Net_0D(num_uniqFeature).cuda()
    # net_grad.load_state_dict(torch.load(path))
    batch_attack_data= np.array([list(GetweightG(input,num_uniqFeature))])
    weight = torch.unsqueeze(torch.tensor(batch_attack_data), dim=1).cuda()
    weight.requires_grad_()
    logit_ = net_grad(weight).cuda()
    loss = CEloss(logit_, torch.LongTensor([label]).cuda())
    # loss = CEloss(logit_, Variable(torch.LongTensor([abs(1-label)])).cuda())
    loss.backward()
    grad = weight.grad.cpu().detach().numpy()[0,0,:]
    logit = logit_[0][int(label)].cpu().detach().numpy()

    return grad,loss.cpu().detach().numpy(),logit

def Getpred(input=[], label=label):
    # net_pred= Net_0D(num_uniqFeature).cuda()
    # net_pred.load_state_dict(torch.load(path))
    batch_attack_data= np.array([list(GetweightG(input,num_uniqFeature))])
    weight = torch.unsqueeze(torch.tensor(batch_attack_data), dim=1).cuda()

    logit = net(weight).cuda()
    pred_label = torch.max(logit[0].cpu().detach(), 0)[1].numpy()
    logit = logit[0][int(label)].cpu().detach().numpy()

    return logit,pred_label
def RSelectCode(topk_set_feature,SetGrad_min_index,Set_c,ori_label):
    SelectCode = random.choice(topk_set_feature)
    Selectset = Set_c[int(SetGrad_min_index[SelectCode])]

    set_att = UnionEle(Selectset, [SelectCode])
    att_data = SetInsert(ori_data, set_att)
    logit,_ = Getpred(input=att_data, label=ori_label)

    return logit,SelectCode,Selectset,set_att

def RSelectCode_lossbase(topk_set_feature,setindex,Set_c,ori_label):
    SelectCode = random.choice(topk_set_feature)
    Selectset = Set_c[setindex]

    set_att = UnionEle(Selectset, [SelectCode])
    att_data = SetInsert(ori_data, set_att)
    logit,_ = Getpred(input=att_data, label=ori_label)

    return logit,SelectCode,Selectset


success_num = 0
success_data = []
danger_data = []
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
    print('========================the number %d/%d========================' % (i, len(label)), file=log_f, flush=True)
    pred_value_ori,pred_label_ori = Getpred(input=ori_data, label=ori_label)
    if pred_label_ori != ori_label:
        NoAttack_num += 1
        print('ori_classifier predicts wrong!', file=log_f, flush=True)
        robust_tag = 0
        continue

    allCode = list(range(num_uniqFeature))
    g_target = [] # The (:the best value of target F function).
    F_S = []  # The (S:the best value of target F function).  it is different from the F_u(S) = F(S+u) - F(S)
    F_value = []

    Set_candidate = []  # the candidate set is S after selecting code process

    Set_target = [()]  # the target set is best chosen attck set_u_att under S
    Set_delet = []

    Set_c = [()]
    iteration = 0
    F_value_index = 0

    while robust_flag == 1:
        iteration +=1
        # print(("----the %dth---")%(iteration), file=log_f, flush=True)

        g_set = []  # this is the list of the value of g(set_u_att).
        code_cand = [] # this is the list of selected code of each set_
        set_u_att = []  # the attack set after selected feature  u under the set_
        F_set = []  # this is the list of the value of F_u(set_u_att). This is to prepare random select
        F_S_new = [] # this is the list of the value of F(set_u_att)

        # Finally we want to save S and its coresponding F_u(S) value F_S
        # save Set_target and its coresponding g_u(S) value g_S
        SetGrad_min = 100*np.ones([num_uniqFeature])
        SetGrad_min_index = np.zeros([num_uniqFeature])
        LOSS = []
        PRED = []
        GRADSET = []
        for set_ in Set_c:
            # get the min_index of categorical and feature under set_ from Set_residual
            set_data_ = SetInsert(ori_data,set_)
            grad_set,loss,logit= GetGradient(input=set_data_, label=ori_label)
            # if logit < pred_value_ori:
            #     print('================', file=log_f, flush=True)
            #     print("prediction", logit, file=log_f, flush=True)
            #     print("loss", loss, file=log_f, flush=True)
            # print('grad: min mean max', [np.min(grad_set),np.mean(grad_set), np.max(grad_set)], file=log_f, flush=True)

            grad_set = abs(grad_set)
            LOSS.append(loss)
            PRED.append(logit)
            GRADSET.append(grad_set)

            SetGrad_min_index = SetGrad_min_index + np.argmin([SetGrad_min, grad_set],axis=0) # the min set index (-1) set compared with former set
            SetGrad_min = np.min([SetGrad_min,grad_set],axis = 0)   # the min grad value for set compared with former set

        grad_sort = np.argsort(SetGrad_min)
        SetGrad_min_index = SetGrad_min_index - 1

        topk_set_feature= DelSortList(DelSortList(list(grad_sort),Set_candidate),Set_delet)[-TopK:]
        pred_value_att,SelectCode,Selectset,set_att= RSelectCode(topk_set_feature,SetGrad_min_index,Set_c,ori_label)

        g_best = np.min(PRED)
        if g_best >= pred_value_att:
            ScoreMin = pred_value_att
            set_att = set_att
        else:
            ScoreMin = g_best
            set_att = Set_c[np.argmin(PRED)]

        F_new = ScoreMin
        F_set = F_new - g_best

        Set_candidate.append(SelectCode)
        Set_target.append(set_att)

        F_S.append(Set_candidate)
        F_S.append(deepcopy(Set_candidate))
        g_target.append(set_att)
        F_value.append(ScoreMin)

        Set_residual = DiffElem(DiffElem(allCode,Set_candidate),Set_delet)
        Set_c = list(powerset(Set_candidate))

        if ScoreMin < TAU:
            success_num += 1
            success_data.append(SetInsert(ori_data,Set_target[-1]))
            danger_data.append(SetInsert(ori_data,Set_target[-2]))
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

    print("Target Score", ScoreMin, file=log_f, flush=True)

    print("  Number of searched codes: %d" % (len(Set_candidate)), file=log_f, flush=True)
    print("  Number of changed codes: %d" % (len(set_att)), file=log_f, flush=True)

    print("  Number of iterations for this: " + str(iteration), file=log_f, flush=True)

    print(" Time: " + str(time.time() - time_start), file=log_f, flush=True)

    print("* SUCCESS Number NOW: %d " % (success_num), file=log_f, flush=True)
    print("* NoAttack Number NOW: %d " % (NoAttack_num), file=log_f, flush=True)

pickle.dump(F,
            open('./AttackCodes/%s/%s_k=%d_t=%s_s=%d_F.pickle' % (MODEL_TYPE,Algo_TYPE, TopK, str(TAU), SECONDS), 'wb'))
pickle.dump(g,
            open('./AttackCodes/%s/%s_k=%d_t=%s_s=%d_g.pickle' % (MODEL_TYPE,Algo_TYPE, TopK, str(TAU), SECONDS), 'wb'))
pickle.dump(F_V, open('./AttackCodes/%s/%s_k=%d_t=%s_s=%d_F_V.pickle' % ( MODEL_TYPE,Algo_TYPE, TopK, str(TAU), SECONDS), 'wb'))
pickle.dump(success_data, open('./AttackCodes/%s/%s_k=%d_t=%s_s=%d_success_data.pickle' % ( MODEL_TYPE,Algo_TYPE, TopK, str(TAU), SECONDS), 'wb'))
pickle.dump(danger_data, open('./AttackCodes/%s/%s_k=%d_t=%s_s=%d_danger_data.pickle' % ( MODEL_TYPE,Algo_TYPE, TopK, str(TAU), SECONDS), 'wb'))


print("--- Total Success Number: " + str(success_num) + " ---", file=log_f, flush=True)
print("--- Total No Attack Number: " + str(NoAttack_num) + " ---", file=log_f, flush=True)
print("--- success Ratio: " + str(success_num/(len(label) - NoAttack_num )) + " ---", file=log_f, flush=True)
print("--- Mean Iteration: " + str(Total_iteration / (len(label) - NoAttack_num)) + " ---", file=log_f, flush=True)
print("--- Mean TargetCode: " + str(Total_targCode / (len(label) - NoAttack_num)) + " ---", file=log_f, flush=True)
print(TITLE)
print(TITLE, file=log_f, flush=True)
