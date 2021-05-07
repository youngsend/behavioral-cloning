import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import *
from dataset import *
from util import *

# Done 1: crop image by top-left, height, width rather than center crop
# Done 2: add left and right images in dataset
# Done 3: add random horizontal flip and modify steer
# ToDo 4: merge data from several csv files (easy, concatenate pandas frame)
# Done 5: split train dataset and validation dataset

# dataset, transforms and data loader
transform = transforms.Compose([
    transforms.Lambda(lambda img: T_F.crop(img, 60, 0, 80, 320)),
    transforms.ToTensor(),
])
dataset = BehaviorCloneDataset(csv_file='data/new_data/driving_log.csv', root_dir='data/new_data', transform=transform)
n_samples = len(dataset)

# split train and validation data
train_size = int(n_samples * 0.9)
val_size = n_samples - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
print('train dataset size {}, val dataset size {}'.format(len(train_dataset), len(val_dataset)))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# model, optimizer, loss function
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LeNetRevised().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# train
train_util = TrainUtil(model, device, loss_fn, train_loader, val_loader)
train_util.training_loop(n_epochs=10, optimizer=optimizer)

# save model
torch.save(model.state_dict(), 'checkpoint/model.pth')
