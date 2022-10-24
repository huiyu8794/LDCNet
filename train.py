from models.LDCN import *
import torch.optim as optim
import itertools
import numpy as np
import os
import random
from pytorch_grad_cam import GradCAM
import torchvision.transforms as T
import matplotlib.pyplot as plt
from loss.hard_triplet_loss import HardTripletLoss

'''
Reference 
Gradcam: https://github.com/jacobgil/pytorch-grad-cam
    
Regularized Fine-Grained Meta Face Anti-Spoofing (AAAI'20)
- https://github.com/rshaojimmy/RFMetaFAS
'''

device_id = 'cuda:0'
batch_size = 5
log_step = 50
model_save_epoch = 1

# Load npy data
dataset1 = "casia"
dataset2 = "MSU" 
dataset3 = "replay"

# image shape: torch.Size([3, 256, 256])
live_path1 = '/shared/domain-generalization/' + dataset1 + '_images_live.npy'
live_path2 = '/shared/domain-generalization/' + dataset2 + '_images_live.npy'
live_path3 = '/shared/domain-generalization/' + dataset3 + '_images_live.npy'

print_path1 = '/shared/domain-generalization/' + dataset1 + '_print_images.npy'
print_path2 = '/shared/domain-generalization/' + dataset2 + '_print_images.npy'
print_path3 = '/shared/domain-generalization/' + dataset3 + '_print_images.npy'

replay_path1 = '/shared/domain-generalization/' + dataset1 + '_replay_images.npy'
replay_path2 = '/shared/domain-generalization/' + dataset2 + '_replay_images.npy'
replay_path3 = '/shared/domain-generalization/' + dataset3 + '_replay_images.npy'

# Path to save activation map
live_amap_path1 = '/shared/huiyu8794/activation_map_results/activation_live_' + dataset1 + '.npy'
live_amap_path2 = '/shared/huiyu8794/activation_map_results/activation_live_' + dataset2 + '.npy'
live_amap_path3 = '/shared/huiyu8794/activation_map_results/activation_live_' + dataset3 + '.npy'

spoof_amap_path1 = '/shared/huiyu8794/activation_map_results/activation_spoof_' + dataset1 + '.npy'
spoof_amap_path2 = '/shared/huiyu8794/activation_map_results/activation_spoof_' + dataset2 + '.npy'
spoof_amap_path3 = '/shared/huiyu8794/activation_map_results/activation_spoof_' + dataset3 + '.npy'

# Path to save training model
model_save_path = '/shared/huiyu8794/BMVC/test_O/'

# Load the pretrained model for producing activation map
Amodel_path = '/shared/huiyu8794/BMVC/test_O/ANet.tar'


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def get_data_loader(data_path="", amap_path="", data_path2="", data_type="live",batch_size=5, shuffle=True, drop_last=True):
    data = None
    live_spoof_label = None
    live_related = None
    spoof_related = None
    material_label = None
    
    if data_type == "live":
        data = np.load(data_path)
        material_label = np.ones(len(data), dtype=np.int64)

        live_spoof_label = np.ones(len(data), dtype=np.int64)
        live_related = np.load(amap_path)
        spoof_related = np.zeros((len(data), 32, 32, 1), dtype=np.float32)
    else:
        print_data = np.load(data_path)
        replay_data = np.load(data_path2)
        data = np.concatenate((print_data, replay_data), axis=0)

        print_lab = np.zeros(len(print_data), dtype=np.int64)
        replay_lab = np.ones(len(replay_data), dtype=np.int64) * 2 
        material_label = np.concatenate((print_lab, replay_lab), axis=0)

        live_spoof_label = np.zeros(len(data), dtype=np.int64)
        live_related = np.zeros((len(data), 32, 32, 1), dtype=np.float32)
        spoof_related = np.load(amap_path)
        
    # dataset
    trainset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(data, (0, 3, 1, 2))),
                                              torch.tensor(np.transpose(live_related, (0, 3, 1, 2))),
                                              torch.tensor(np.transpose(spoof_related, (0, 3, 1, 2))),
                                              torch.tensor(live_spoof_label),
                                              torch.tensor(material_label))
    # free memory
    import gc
    del data
    del live_related
    del spoof_related
    gc.collect()

    # dataloader
    data_loader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              drop_last=drop_last)
    return data_loader


def get_inf_iterator(data_loader):
    while True:
        for images, live_map, spoof_map, live_spoof_labels, material_label in data_loader:
            yield (images, live_map, spoof_map, live_spoof_labels, material_label)


def produce_sample_level_amap(model=None, image_path="",image_path2="live", save_path="", target_category=None):
    image_data = None
    if image_path2=="live":
        image_data = np.load(image_path)
    else:
        print_data = np.load(image_path)
        replay_data = np.load(image_path2)
        image_data = np.concatenate((print_data, replay_data), axis=0) 

    image_data = torch.tensor(np.transpose(image_data, (0, 3, 1, 2)))
    trainset = torch.utils.data.TensorDataset(image_data)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=False)
    activation_map = np.zeros((len(image_data), 32, 32, 1), dtype=np.float32)
    a_index = 0

    for i, data in enumerate(trainloader, 0):
        image = data[0]
        image = image.to(device_id)
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        grayscale_cam = cam(input_tensor=image, target_category=target_category)
        grayscale_cam = T_transform_resize_32(torch.tensor(grayscale_cam))
        grayscale_cam = grayscale_cam.unsqueeze(3)

        for j in range(image.size(0)):
            activation_map[a_index] = grayscale_cam[j].cpu().detach().numpy() * 255
            a_index += 1

    np.save(save_path, activation_map)


T_transform_resize_32 = T.Resize(32)

AMapModel = ResNet_Amodel().to(device_id)
AMapModel.load_state_dict(torch.load(Amodel_path, map_location=device_id))
AMapModel.eval()
target_layers = [AMapModel.FeatExtor_LS.layer4[-1]]
cam = GradCAM(model=AMapModel, target_layers=target_layers, use_cuda=True)
     
produce_sample_level_amap(model=AMapModel, image_path=live_path1, save_path=live_amap_path1, target_category=1)
produce_sample_level_amap(model=AMapModel, image_path=live_path2, save_path=live_amap_path2, target_category=1)
produce_sample_level_amap(model=AMapModel, image_path=live_path3, save_path=live_amap_path3, target_category=1)
produce_sample_level_amap(model=AMapModel, image_path=print_path1, image_path2=replay_path1, save_path=spoof_amap_path1, target_category=0)
produce_sample_level_amap(model=AMapModel, image_path=print_path2, image_path2=replay_path2, save_path=spoof_amap_path2, target_category=0)
produce_sample_level_amap(model=AMapModel, image_path=print_path3, image_path2=replay_path3, save_path=spoof_amap_path3, target_category=0)

model = LDCNet().to(device_id)
criterionCls =  nn.CrossEntropyLoss().to(device_id)
criterionMSE = torch.nn.MSELoss().to(device_id)
criterionTrip = HardTripletLoss(margin=0.1, hardest=True).to(device_id)
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
model.train()

save_index = 0

data1_real = get_data_loader(data_path=live_path1, amap_path=live_amap_path1, data_type="live",
                             batch_size=batch_size, shuffle=True)
data2_real = get_data_loader(data_path=live_path2, amap_path=live_amap_path2, data_type="live",
                             batch_size=batch_size, shuffle=True)
data3_real = get_data_loader(data_path=live_path3, amap_path=live_amap_path3, data_type="live",
                             batch_size=batch_size, shuffle=True)
data1_fake = get_data_loader(data_path=print_path1, data_path2=replay_path1, amap_path=spoof_amap_path1, data_type="spoof",
                             batch_size=batch_size, shuffle=True)
data2_fake = get_data_loader(data_path=print_path2, data_path2=replay_path2, amap_path=spoof_amap_path2, data_type="spoof",
                             batch_size=batch_size, shuffle=True)
data3_fake = get_data_loader(data_path=print_path3, data_path2=replay_path3, amap_path=spoof_amap_path3, data_type="spoof",
                             batch_size=batch_size, shuffle=True)

iternum = max(len(data1_real), len(data2_real),
              len(data3_real), len(data1_fake),
              len(data2_fake), len(data3_fake))

data1_real = get_inf_iterator(data1_real)
data2_real = get_inf_iterator(data2_real)
data3_real = get_inf_iterator(data3_real)
data1_fake = get_inf_iterator(data1_fake)
data2_fake = get_inf_iterator(data2_fake)
data3_fake = get_inf_iterator(data3_fake)


for epoch in range(500):

    for step in range(iternum):

        # ============ one batch extraction ============ # 
        img1_real, live_img1_liveMap, spoof_img1_liveMap, ls_lab1_real, m_lab1_real = next(data1_real)
        img1_fake, live_img1_fakeMap, spoof_img1_fakeMap, ls_lab1_fake, m_lab1_fake = next(data1_fake)

        img2_real, live_img2_liveMap, spoof_img2_liveMap, ls_lab2_real, m_lab2_real = next(data1_real)
        img2_fake, live_img2_fakeMap, spoof_img2_fakeMap, ls_lab2_fake, m_lab2_fake = next(data1_fake)

        img3_real, live_img3_liveMap, spoof_img3_liveMap, ls_lab3_real, m_lab3_real = next(data1_real)
        img3_fake, live_img3_fakeMap, spoof_img3_fakeMap, ls_lab3_fake, m_lab3_fake = next(data1_fake)

        # ============ one batch collection ============ # 

        catimg1 = torch.cat([img1_real, img1_fake], 0).to(device_id)
        live_amap1 = torch.cat([live_img1_liveMap, live_img1_fakeMap], 0).to(device_id)
        spoof_amap1 = torch.cat([spoof_img1_liveMap, spoof_img1_fakeMap], 0).to(device_id)
        ls_lab1 = torch.cat([ls_lab1_real, ls_lab1_fake], 0).to(device_id)
        m_lab1 = torch.cat([m_lab1_real, m_lab1_fake], 0).to(device_id)

        catimg2 = torch.cat([img2_real, img2_fake], 0).to(device_id)
        live_amap2 = torch.cat([live_img2_liveMap, live_img2_fakeMap], 0).to(device_id)
        spoof_amap2 = torch.cat([spoof_img2_liveMap, spoof_img2_fakeMap], 0).to(device_id)
        ls_lab2 = torch.cat([ls_lab2_real, ls_lab2_fake], 0).to(device_id)
        m_lab2 = torch.cat([m_lab2_real, m_lab2_fake], 0).to(device_id)

        catimg3 = torch.cat([img3_real, img3_fake], 0).to(device_id)
        live_amap3 = torch.cat([live_img3_liveMap, live_img3_fakeMap], 0).to(device_id)
        spoof_amap3 = torch.cat([spoof_img3_liveMap, spoof_img3_fakeMap], 0).to(device_id)
        ls_lab3 = torch.cat([ls_lab3_real, ls_lab3_fake], 0).to(device_id)
        m_lab3 = torch.cat([m_lab3_real, m_lab3_fake], 0).to(device_id)

        # ============ concatenate three datasets to list ============ # 

        catimglist = torch.cat([catimg1, catimg2, catimg3], 0)
        liveGT_list = torch.cat([live_amap1, live_amap2, live_amap3], 0)
        spoofGT_list = torch.cat([spoof_amap1, spoof_amap2, spoof_amap3], 0)
        ls_lab_list = torch.cat([ls_lab1, ls_lab2, ls_lab3], 0)
        m_lab_list = torch.cat([m_lab1, m_lab2, m_lab3], 0)

        # ============ random sample data from lists ============ # 

        batchidx = list(range(len(catimglist)))
        random.shuffle(batchidx)

        img_rand = catimglist[batchidx, :]
        liveGT_rand = liveGT_list[batchidx, :]
        spoofGT_rand = spoofGT_list[batchidx, :]
        ls_lab_rand = ls_lab_list[batchidx]
        m_lab_rand = m_lab_list[batchidx]

        pred, live_amap_pred, spoof_amap_pred, features = model(img_rand) 

        Loss_cls = criterionCls(pred.squeeze(), ls_lab_rand)
        Loss_dep = criterionMSE(live_amap_pred, liveGT_rand)
        Loss_dep += criterionMSE(spoof_amap_pred, spoofGT_rand)
        Loss_trip = criterionTrip(features, m_lab_rand)
        
        Loss_all = Loss_cls + Loss_dep * 0.004 + Loss_trip * 0.1
        
        optimizer.zero_grad()
        Loss_all.backward()
        optimizer.step()

        if (step + 1) % log_step == 0:
            print('[epoch %d step %d]  Loss_cls %.9f  Loss_dep %.9f Loss_tri %.9f'
                  % (epoch, step, Loss_cls.item(), Loss_dep.item() * 0.004, Loss_trip.item() * 0.1))

        
    if ((epoch + 1) % model_save_epoch == 0):
        mkdir(model_save_path)
        save_index += 1
        torch.save(model.state_dict(), os.path.join(model_save_path,
                                                      "LDCNet-{}.tar".format(save_index))) 
