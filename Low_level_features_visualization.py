from models.LDCN_low_level_visualization import *
import numpy as np
import os 
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

device_id = "cuda:0"

dataset = 'Oulu'
live_path = '/shared/domain-generalization/' + dataset + '_images_live.npy'
spoof_path = '/shared/domain-generalization/' + dataset + '_images_spoof.npy'

image_save_path = "./low_level_visualize/" + dataset + "_example/"

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def imshow_np(img, filename, image_save_path=""):
    height, width, depth = img.shape
    if depth == 1:
        img = img[:, :, 0]
        plt.imshow(img, cmap=plt.get_cmap('gray'))
    else:
        plt.imshow(img)
    plt.axis("off")
    plt.savefig(image_save_path + filename + ".png",
                bbox_inches='tight', pad_inches=0)
    plt.close()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


live_data = np.load(live_path)
spoof_data = np.load(spoof_path)

live_label = np.ones(len(live_data), dtype=np.int64)
spoof_label = np.zeros(len(spoof_data), dtype=np.int64)

total_data = np.concatenate((live_data, spoof_data), axis=0)
total_label = np.concatenate((live_label, spoof_label), axis=0)

trainset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(total_data, (0, 3, 1, 2))),
                                          torch.tensor(total_label))
# dataloader
data_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

mkdir(image_save_path)

model_path = "./pretrained_model/LDCN_for_visualize.tar"
model = LDCNet_visualize().to(device_id) 
model.load_state_dict(torch.load(model_path, map_location=device_id))
model.eval()

for i, data in enumerate(data_loader, 0):
    images, labels = data
    images = images.to(device_id)
    pred, live_pred, spoof_pred, dx1, dx2 = model(images)
    
    # Low-level feature
    dx2 = torch.sum(dx2, 1, True) # dx2.size() = [2, 128, 128, 128]

    # draw original image
    imshow_np(np.transpose(images[0].detach().cpu().numpy() * 255, (1, 2, 0)),
                str("{0:04d}_image".format(i)) + str("_{0:01d}".format(labels[0])), image_save_path)

    # draw low-level feature map
    imshow_np(np.transpose(1-NormalizeData(dx2[0].detach().cpu().numpy()), (1, 2, 0)),
                str("{0:04d}_featmap".format(i)), image_save_path)

    print(image_save_path +str("{0:04d}_featmap".format(i)) + ' saved')
    
    