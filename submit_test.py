import torch
import os
import re
import torch.nn.functional as F
import numpy as np
import random
from model import build_resunetplusplus_binary
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_size=16

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

class GetLoader2(torch.utils.data.Dataset):
    def __init__(self, slide_id="111",color=1):
        self.save_original = "/media/lingang/bingli-1/fenji_segmentation_64/savedata_data_downsample_pair/"
        data_names = np.load("./binary_results/data_filenames5_"+str(color)+".npy")
        self.color = color
        x_filenames = []
        for data_name in data_names:
            if 'mask' not in data_name:
                data_name = re.split('/', data_name)[-1]
                if re.split('_', data_name)[0] == slide_id:
                    x_filenames.append(self.save_original + data_name)
        self.dataset = x_filenames
        self.n_data = len(x_filenames)

    def __getitem__(self, item):
        x_data = np.load(self.dataset[item])/255
        #x_data = getnormalization(x_data)
        y_data = np.load(self.dataset[item][:-4]+"_mask.npy")
        y_data = y_data[self.color, :, :]
        x_data = torch.from_numpy(x_data)
        y_data = torch.from_numpy(y_data)
        return x_data, y_data, self.dataset[item]

    def __len__(self):
        return self.n_data

def load_data_test1(slide_id="111",color=1):
    dataset1 = GetLoader2(slide_id=slide_id,color=color)
    dataloader1 = torch.utils.data.DataLoader(dataset=dataset1,batch_size=batch_size,shuffle=False)
    return dataloader1

def testset(model,color,threshold=0.5):
    filenames = np.load("./data_filenames.npy")
    data_keys = [re.split('/', filename)[-1][:-4] for filename in filenames]

    x_test = ["150233A1A2A3-LPA80-APA20", "150008A1A2-PPA70-APA20-CGP10", "155516A3A4", "150508A1A2A3-MPA85-CGP10-PPA5",
              "150521A1A2A3-CGP90-SPA10", "151999A1A2A3-APA90-MPA10"]
    x_train = [x for x in data_keys if x not in x_test]
    save_path = "./Figures_pred/model_binary/resnet/" + str(color) + "/results/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.eval()
    for slide_id in x_test:
        test_loader=load_data_test1(slide_id=slide_id,color=color)

        true_value = np.zeros((1, 1, 256, 256))
        pred_value = np.zeros((1, 1, 256, 256))
        x_filenames=[]
        with torch.no_grad():
            for i, pack in enumerate(test_loader):
                x_filenames.extend(pack[2])
                image, gt = pack[0].type(torch.FloatTensor), pack[1].type(torch.LongTensor)
                gt = gt.cuda()
                image = image.cuda()
                image = image.permute(0, 3, 1, 2)
                gt = gt.unsqueeze(dim=1)
                res = model(image)
                true_value = np.concatenate((true_value, gt.cpu().numpy()))
                y_pred = F.sigmoid(res)
                y_pred = (y_pred > threshold).float()
                pred_value = np.concatenate((pred_value, y_pred.cpu().numpy()))

        true_value = true_value[1:]
        pred_value = pred_value[1:]
        print('start plot')
        np.save(save_path + slide_id + "_pred.npy",pred_value.reshape((-1,1,256,256)))
        np.save(save_path + slide_id + "_true.npy", true_value.reshape((-1,1,256,256)))
        np.save(save_path + slide_id + "_filenames.npy", np.array(x_filenames))

def init_weight(net,restore):
    net.load_state_dict(torch.load(restore))
    print("Restore model from: {}".format(restore))
    return net

def get_best_model_name(filename):
    pathDirs=os.listdir(filename)
    index_list=[]
    best_filename=''
    for pathDir in pathDirs:
        index_list.append(float(re.split('_',pathDir)[0]))
    if len(index_list)>0:
        a=index_list.index(max(index_list))
        best_filename=filename+pathDirs[a]
    else:
        print(filename+' no pth!')
    return best_filename

if __name__ == '__main__':
    colors=[1,2,3,4,5,6]
    labels = ['other', '255', "16711680", "8454016", "16711935", "0", "65535"]
    contents1=[]
    for color in colors:
        # seed constant
        seed_torch(seed=111)
        # ---- build models ----
        model = build_resunetplusplus_binary().cuda()
        save_filename = get_best_model_name("./model_binary/resnet/" + str(color) + "/")
        model = init_weight(model, save_filename)
        print(labels[colors.index(color) + 1])
        testset(model, color)