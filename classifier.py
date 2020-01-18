import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x,dim=1)


classify = ConvNet()
optimizer = optim.SGD(classify.parameters(), lr=0.01, momentum=0.5)

losses = []
wins = []


def train (epoch, trainer):
    # stuff
    classify.train()

    for batch_id, (data, labels) in enumerate(trainer):
        data = Variable(data)
        target = Variable(labels)

        optimizer.zero_grad()
        preds = classify(data)
        loss = F.nll_loss(preds,target)
        #print(loss.data[0])
        loss.backward()
        losses.append(loss.item())

        optimizer.step()

        if batch_id % 1000 == 0:
            print(loss.item())


def test(epoch, tester):
    with torch.no_grad():
        classify.eval()

        test_loss = 0
        win_count = 0

        for data,target in tester:
            data = Variable(data)
            target = Variable(target)

            out = classify(data)
            test_loss += F.nll_loss(out, target).item()
            #print(test_loss)
            pred = out.data.max(1)[1]
            win_count += pred.eq(target.data).cpu().sum()

        test_loss = test_loss
        test_loss /= len(tester)

        win_perc = 100 * win_count/len(tester.dataset)
        wins.append(win_perc)

        print(f"Average loss: {test_loss} ...... Average Correct: {win_perc}")


train_load = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_load = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# Separate between training set and test set
trainset = torch.utils.data.DataLoader(train_load, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test_load, batch_size=10, shuffle=True)

for epoch in range(0, 3):
    print(f"Epoch: {epoch}")

    train(epoch, trainset)
    test(epoch, testset)








