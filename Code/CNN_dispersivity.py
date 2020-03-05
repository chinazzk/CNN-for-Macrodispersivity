import torch 
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import torchvision.transforms as transforms
from utils.load_data import load_data
from utils.misc import mkdirs
from utils.plot import save_stats
from args_cnn import args
import numpy as np
from sklearn.metrics import r2_score
from time import time
import json   # 
import os # 
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'  # 超算
args.train_dir = args.run_dir + "/training"
args.pred_dir = args.train_dir + "/predictions"
mkdirs([args.train_dir, args.pred_dir])

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=1), #
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))                    #   
        #self.conv2_drop1 = nn.Dropout2d(0.5)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=2),     # 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))                     # 
        #self.conv2_drop2 = nn.Dropout2d(0.2)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=2), # 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))                 # 16 > 8
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=2), # 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))                 # 16 > 8
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2), # 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))                 # 16 > 8
        #self.conv2_drop3 = nn.Dropout2d(0.8)
        self.fc1 = nn.Linear(256,256)  #dim*256 ->256*120  #uu'chu
        #self.avg_pool = nn.AvgPool2d(8)
        #self.bat1 = nn.BatchNorm1d(784)
        self.relu1 = nn.ReLU()
        #self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256,256)
        #self.bat2 = nn.BatchNorm1d(784)
        self.relu2 = nn.ReLU()
        #self.dropout2 = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes) # 100
        
    def forward(self, x):
        out = self.layer1(x)
        #out = self.conv2_drop1(out) # add for conv2d
        out = self.layer2(out)
        #out = self.conv2_drop2(out) # add for conv2d
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        #out = self.conv2_drop3(out) # add for conv2d
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        #out = self.bat1(out)
        out = self.relu1(out)
        #out = self.dropout1(out)
        out = self.fc2(out)
        #out = self.bat2(out)
        out = self.relu2(out)
        #out = self.dropout2(out)
        out = self.fc(out)
        return out
# initialize model
model = ConvNet(args.num_class).to(device)
#print(model)

# load checkpoint if in post mode 
if args.post:
    checkpoint = args.ckpt_dir + '/model_epoch{}.pth'.format(args.ckpt_epoch)
    model.load_state_dict(torch.load(checkpoint))
    print('Loaded pre-trained model: {}'.format(checkpoint))

# load data
train_data_dir = args.data_dir + '/V{}N{}L{}_no_chunk_train.hdf5'.format(args.var, args.ntrain, args.cor)

test_data_dir = args.data_dir +'/V{}N{}L{}_no_chunk_train.hdf5'.format(args.var, args.ntest, args.cor)
train_loader, train_stats = load_data(train_data_dir, args.batch_size)
test_loader, test_stats = load_data(test_data_dir, args.test_batch_size)
print('Loaded data!')

# Loss and optimizer
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                            weight_decay=args.weight_decay) # fix lr



scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                verbose=True, threshold=0.0001, threshold_mode='rel',
                                cooldown=0, min_lr=0, eps=1e-8)


logger = {}
logger['rmse_train'] = []
logger['rmse_test'] = []
logger['r2_train'] = []
logger['r2_test'] = []

# Train the model
def train(epoch):
    model.train()
    mse = 0.
    for i,(images,labels) in enumerate(train_loader):
        images,labels = images.to(device), labels.to(device)
        model.zero_grad()  # add by zhou 每轮batch 清零 梯度 不然就是变相增加batch？
        outputs = model(images)
        outputs = torch.squeeze(outputs) # add by zhou (16,1)->(16)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        mse += loss.item()

    rmse = np.sqrt(mse/args.ntrain)

    r2_score = 1- mse/ train_stats['y_var']
    print("epoch: {}, training r2-score: {:.6f}, training rmse: {:.6f}".format(epoch, r2_score, rmse))
    if epoch % args.log_freq == 0:
        logger['r2_train'].append(r2_score)
        logger['rmse_train'].append(rmse)
    # save model
    if epoch % args.ckpt_freq == 0:
        torch.save(model.state_dict(), args.ckpt_dir + "/model_epoch{}.pth".format(epoch))
# eval the model
def test(epoch):
    model.eval()
    mse = 0.
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        outputs = torch.squeeze(outputs)
        mse += criterion(outputs,labels).item()

        # plot predictions
        if epoch % args.plot_freq == 0 and batch_idx == 0:
            n_samples = 50 if epoch == args.epochs else 2 # 
            idx = torch.randperm(images.size(0))[:n_samples] # 
            samples_outputs = outputs.data.cpu()[idx].numpy()
            samples_images = labels.data.cpu()[idx].numpy()
            print('epoch {}: reference {} prediction {}'.format(epoch,samples_images,
                                                            samples_outputs))


    rmse_test = np.sqrt(mse/ args.ntest)
    scheduler.step(rmse_test)
    r2_score = 1- mse /test_stats['y_var']
    print("epoch: {}, test r2-score: {:.6f}, testing rmse: {:.6f}".format(epoch, r2_score, rmse_test))

    if epoch % args.log_freq == 0:
        logger['r2_test'].append(r2_score)
        logger['rmse_test'].append(rmse_test)

# begin training
print('start training......................')
tic = time()
for epoch in range(1, args.epochs + 1):
    train(epoch)
    with torch.no_grad(): # 
        test(epoch)
tic2 = time()
print("Finished training {} epochs with {} data using {} seconds (including long... plotting time)"
        .format(args.epochs,args.ntrain,tic2-tic))

# plot 
x_axis = np.arange(args.log_freq, args.epochs + args.log_freq, args.log_freq) # 2,202,2
save_stats(args.train_dir,logger,x_axis)

args.training_time = tic2 - tic

with open(args.run_dir + "/args.txt", 'w') as args_file:
    json.dump(vars(args), args_file, indent=4)
