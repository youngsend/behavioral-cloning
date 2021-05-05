import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import *
from dataset import *
from util import *

# dataset, transforms and data loader
transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = BehaviorCloneDataset(csv_file='data/driving_log.csv', root_dir='data', transform=transform)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# model, optimizer, loss function
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LeNetRevised().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

training_loop(n_epochs=10, optimizer=optimizer, model=model, loss_fn=loss_fn, train_loader=train_loader,
              device=device)

torch.save(model.state_dict(), 'checkpoint/model.pth')
