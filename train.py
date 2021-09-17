import argparse
from models import ResNet_custom,NLNet,SENet,Self_attention 
import torch.nn as nn
from data_loader import pytorch_data_loader
import torchvision
import torch

# Put all the models and the inputs on the cuda device if available 

def train_model(model,train_dataloader,criterion,optimizer,device=torch.device("cuda:0")):

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            labels = labels.type(torch.LongTensor)
            labels = torch.squeeze(labels,dim=-1)

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            outputs = outputs.to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

    print('Finished Training')

    return 0


def test_acc(model,test_dataloader,device=torch.device("cuda:0")):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            labels = labels.type(torch.LongTensor)

            images = images.to(device)
            labels = labels.to(device)

            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)                                 # Add up the batch sizes here 
            correct += (predicted == labels).sum().item()           # See the correct labels here 

    print('Accuracy of the network on the 823 test images: %d %%' % (
        100 * correct / total))

    return 0


def main(args):

    # We first extract the Pytorch Dataloader here 
    print("\n Loading flowers data into pytorch dataloader \n")

    trainloader,testloader = pytorch_data_loader.train_test_dataloader(args.data_path)



    # We test all the models one by one here    ResNet34 custom: 
    print("\nCustom ResNet34 module \n")

    resnet = ResNet_custom.ResNet()
    resnet = nn.Sequential(resnet)


    resnet.apply(ResNet_custom.weights_init)
    resnet = resnet.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)

    print("\nPrinting Training Loss\n")

    train_model(resnet,trainloader,criterion,optimizer)


    print("\nPrinting Test Accuracy\n")

    test_acc(resnet,testloader)

    # # Pytorch Builtin ResNet module 

    print("\nPytorch ResNet34 module \n")

    model = torchvision.models.resnet34(pretrained=False)
    model = nn.Sequential(model,nn.Linear(1000,5))                # 5 output classes in here :: 
    model = model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nPrinting Training Loss\n")

    train_model(model,trainloader,criterion,optimizer)

    print("\nPrinting Test Accuracy\n")

    test_acc(model,testloader)


    # # Custom SENet module 

    print("\nCustom SENet module  \n")

    senet = SENet.SENet()
    senet = nn.Sequential(senet)



    senet.apply(SENet.weights_init)

    senet = senet.to(args.device)



    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(senet.parameters(), lr=0.001)

    print("\nPrinting Training Loss\n")

    train_model(senet,trainloader,criterion,optimizer)



    print("\nPrinting Test Accuracy\n")

    test_acc(senet,testloader)


    # # Custom NLNet module 

    print("\nCustom NLNet module  \n")

    resnet_nlblock = NLNet.ResNet_NLBlock()                      # Non local block added to one of the sub-blocks in the resnet architecture here 
    resnet_nlblock = nn.Sequential(resnet_nlblock)
    resnet_nlblock.apply(NLNet.weights_init)
    resnet_nlblock = resnet_nlblock.to(args.device)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet_nlblock.parameters(), lr=0.001)

    print("\nPrinting Training Loss\n")

    train_model(resnet_nlblock,trainloader,criterion,optimizer)

    print("\nPrinting Test Accuracy\n")

    test_acc(resnet_nlblock,testloader)


    # # Custom Attention module 

    print("\nCustom Attention module \n")

    sa_resnet = Self_attention.SA_ResNet()
    sa_resnet = nn.Sequential(sa_resnet)

    sa_resnet.apply(Self_attention.weights_init) 

    sa_resnet = sa_resnet.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(sa_resnet.parameters(), lr=0.001)

    print("\nPrinting Training Loss\n")

    train_model(sa_resnet,trainloader,criterion,optimizer)

    print("\nPrinting Test Accuracy\n")

    test_acc(sa_resnet,testloader)









if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Some CV models tested on the flowers dataset')

    parser.add_argument('--device', default=torch.device("cuda:0"))

    parser.add_argument('--data_path', type=str, default='/home/bhushan/Desktop/bhushan_env/bhushan/Attention in CV/attn_cv_2/output_hdf5_flowers/data.h5', help='HDF5 file path')


    args, unknown = parser.parse_known_args()              # To Run this in Jupyter 

    args = parser.parse_args()                             # To Run it in Python 

    main(args)

