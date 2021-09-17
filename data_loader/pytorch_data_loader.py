
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import h5py
from PIL import Image
from torchvision import datasets
from torchvision.transforms import ToTensor



# data loader function here: 
class ImageDataset(Dataset):

    def __init__(self, datasetFile, transform=None):
        self.datasetFile = datasetFile
        self.transform = transform
        self.dataset = None
        self.dataset_keys = None
        self.h5py2int = lambda x: int(np.array(x))

    def __len__(self):
        f = h5py.File(self.datasetFile, 'r')
        self.dataset_keys = [str(k) for k in f.keys()]       # it's already string but still convert it here 
        length = len(f)//2           # data and labels so twice here 
        f.close()

        return length

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.datasetFile, mode='r')
            self.dataset_keys = [str(k) for k in self.dataset.keys()]
            
        # Image and label here :
        
        data_key = 'X'+str(idx) 
        label_key = 'y'+str(idx)
        
        image = self.dataset[data_key][()]           # Numpy array
#         print(image.dtype)
        image = Image.fromarray(image)    # PIL image here 
        label = self.dataset[label_key][()]
        
        # Transform here : 
        
        image = self.transform(image)
        return (image,label)




def train_test_dataloader(hdf5_path="D:\\Attention in Computer Vision\\output_hdf5_flowers\\data.h5",data_mean=torch.tensor([0.4582, 0.4185, 0.3001]),data_std=torch.tensor([0.2940, 0.2640, 0.2845])):
    """
        Take in the dataset path for the hdf5 file and return the pytorch train and test loaders 
        Our image dataset is already resized so we do not do it here 
    """
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std)
    ])

    im_dataset = ImageDataset(hdf5_path,transformation)

    train_dataset,test_dataset = torch.utils.data.random_split(im_dataset, [3500, 823])

    train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=True)

    return train_dataloader,test_dataloader


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))       # NCHW followed by pytorch here :: 
    plt.show()


# trainloader,testloader = train_test_dataloader()

# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# imshow(torchvision.utils.make_grid(images))
# print(' '.join('%5s' % [labels[j]] for j in range(32)))













