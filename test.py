from models.LDCN import *
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score

device_id = 'cuda:0'
model_path = '/shared/huiyu8794/BMVC/test_O/'
batch_size = 40

dataset = 'Oulu'
live_path = '/shared/domain-generalization/' + dataset + '_images_live.npy'
spoof_path ='/shared/domain-generalization/' + dataset + '_images_spoof.npy'

def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    # y = TPR - FPR
    y = TPR + (1 - FPR)
    # print(y)
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


live_data = np.load(live_path)
spoof_data = np.load(spoof_path)

live_label = np.ones(len(live_data), dtype=np.int64)
spoof_label = np.zeros(len(spoof_data), dtype=np.int64)

total_data = np.concatenate((live_data, spoof_data), axis=0)
total_label = np.concatenate((live_label, spoof_label), axis=0)

trainset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(total_data, (0, 3, 1, 2))),
                                          torch.tensor(total_label))

# dataloader
data_loader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=False)

model = LDCNet().to(device_id)

for epoch in range(1, 500):
    MataNet_path = model_path + 'LDCNet-' +str(epoch) + ".tar"

    model.load_state_dict(torch.load(MataNet_path, map_location=device_id))
    model.eval()

    score_list = []
    score_list_live = []
    score_list_spoof = []
    Total_score_list_cs = []
    label_list = []
    TP = 0.0000001
    TN = 0.0000001
    FP = 0.0000001
    FN = 0.0000001
 
    for i, data in enumerate(data_loader, 0):
        # print(i)
        images, labels = data
        images = images.to(device_id)
        label_pred, _, _, _ = model(images)
        score = F.softmax(label_pred, dim=1).cpu().data.numpy()[:, 1]

        for j in range(images.size(0)):
            score_list.append(score[j]) 
            label_list.append(labels[j])
 
    score_list = NormalizeData(score_list)

    fpr, tpr, thresholds_cs = metrics.roc_curve(label_list, score_list)
    threshold_cs, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds_cs)
 

    for i in range(len(score_list)):
        score = score_list[i]
        if (score >= threshold_cs and label_list[i] == 1):
            TP += 1
        elif (score < threshold_cs and label_list[i] == 0):
            TN += 1
        elif (score >= threshold_cs and label_list[i] == 0):
            FP += 1
        elif (score < threshold_cs and label_list[i] == 1):
            FN += 1
 
    APCER = FP / (TN + FP)
    NPCER = FN / (FN + TP) 

    print('[epoch %d]  ACER %.4f  AUC %.4f'
            % (epoch, np.round((APCER + NPCER) / 2, 4), roc_auc_score(label_list, score_list)))