import datetime
import torch


class TrainUtil:
    """Training utility class including train loop and validation loss calculation."""
    def __init__(self, model, device, loss_fn, train_loader, val_loader):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader

    def validate(self, loader):
        self.model.eval()
        loss_total = 0.0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=self.device)
                labels = labels.unsqueeze(1).to(device=self.device)
                outputs = self.model(imgs)
                loss = self.loss_fn(outputs, labels)
                loss_total += loss.item()

        print("val loss: {}".format(loss_total / len(loader)))

    def training_loop(self, n_epochs, optimizer):
        self.model.train()  # needed when using dropout, batchnorm etc.
        for epoch in range(1, n_epochs + 1):
            loss_train = 0.0
            for imgs, labels in self.train_loader:
                imgs = imgs.to(device=self.device)
                # because this warning: Using a target size (torch.Size([64])) that is different to the input size
                # (torch.Size([64, 1])).
                labels = labels.unsqueeze(1).to(device=self.device)
                outputs = self.model(imgs)
                loss = self.loss_fn(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
            if epoch == 1 or epoch % 1 == 0:
                print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(),
                                                             epoch, loss_train / len(self.train_loader)))
                # compare train loss and validation loss
                self.validate(self.val_loader)
