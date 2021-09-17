import torch
import torch.nn as nn
import torch.nn.functional as F


# Define a residual network here :: 
# 34 layer residual network here :: fixed architecture here 
# We define the fundamental block architecture here :: constructs the intermediate blocks 

# kernel size is by default 3 for all the blocks 
# The first layer in every block has stride 2 so that halving of the image size occurs here 
# Skip connections go across the subblocks here :: 
# Batch Normalization after each layer needs to be done here 
# The in_channels of the current sub_block depend upon the previous sub_block out_channels :: or previous sub block's num_filters 
# Number of filters in a layer is nothing but the out channels of that layer/block 
# All the conv layers in a sub_block have the same number of filters 
# Pytorch follows N C H W convention N : batchsize 
# Relu needs to be specified explicitly BatchNorm before ReLu :: Important here 

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)



class first_sub_block(nn.Module):               # The first layer of the first sub-block reduces the size 
    
    def __init__(self,in_channels,kernel_size,num_filters):
        
        super().__init__() 
        self.conv1 = nn.Conv2d(in_channels,num_filters,kernel_size,stride=2,padding=kernel_size//2)   # reduce by 1/2
        self.conv2 = nn.Conv2d(num_filters,num_filters,kernel_size,stride=1,padding=kernel_size//2)   # keep the same size 
        self.batchnorm1 = nn.BatchNorm2d(num_filters)         # In each layer the num_filters is fixed :: for a block 
        self.batchnorm2 = nn.BatchNorm2d(num_filters)
        self.batchnorm3 = nn.BatchNorm2d(num_filters)
        self.ReLu = nn.ReLU()
        self.conv1x1 = nn.Conv2d(in_channels,num_filters,1,stride=2,padding=0)
        
    def forward(self,x): 
        
        # The first sub-block :: first layer halves the size followed by the same size conv layer 
        
        x_skip = self.conv1x1(x)            # Change the number of filters as well as the size here 
        x_skip = self.batchnorm3(x_skip)    # Here too put a batchnorm 
        
        x1 = self.conv1(x)                     # resize by 1/2 
        x1 = self.batchnorm1(x1)
        x1 = self.ReLu(x1)

        x2 = self.conv2(x1)                    # same size here 

        x2 = self.batchnorm2(x2)


        out = x2 + x_skip                          # Residual connection added here 

        out = self.ReLu(out)
        
        
        return out 

class sub_block(nn.Module):                                    # Size is maintained here 
    
    def __init__(self,num_filters,kernel_size):                # The sub-blocks DO NOT halve the activation map hence stride and padding are default 
        
        # Each block has a subblock which is replicated through the architecture in the Block 
        # The first sub-block is put in the block itself 
        super().__init__()
        
        self.conv1 = nn.Conv2d(num_filters,num_filters,kernel_size,stride=1,padding=kernel_size//2)    # Only one conv layer here 
        self.conv2 = nn.Conv2d(num_filters,num_filters,kernel_size,stride=1,padding=kernel_size//2)
        self.batchnorm1 = nn.BatchNorm2d(num_filters)         # In each layer the num_filters is fixed :: for a block 
        self.batchnorm2 = nn.BatchNorm2d(num_filters) 
        self.ReLu = nn.ReLU()
        
    def forward(self,x):                                     # Traverse through a sub-block here :: 
        
        x1 = self.conv1(x)                 # doesn't change the size here 
        x1 = self.batchnorm1(x1)
        x1 = self.ReLu(x1)


        x2 = self.conv2(x1)                # doesn't change the size here 
        x2 = self.batchnorm2(x2)


        out = x + x2          # Finally the output of the block would be computed here 

        out = self.ReLu(out)
        
        return out 



class Block(nn.Module):
    def __init__(self,in_channels,kernel_size,num_filters,num_sub_blocks):
        super().__init__()
        self.num_iterations = num_sub_blocks                 # Skip connections across sub_blocks here 
#         self.ReLu = nn.ReLU()
        
        
        self.sub_block_list = nn.ModuleList([first_sub_block(in_channels,kernel_size,num_filters)])          # Added separately here :: 
        
        
        for i in range(self.num_iterations-1):
            self.sub_block_list.append(sub_block(num_filters,kernel_size))       # Normal sub-blocks append to the end of the list 
        
        
         
        
        
    def forward(self,x):                             # Remember the skip connections across the sub_blocks 
                
        out = x                                  # Input would be traversed across all the layers here 
        
        for i in range(self.num_iterations):
            out = self.sub_block_list[i](out)
        return out                                # Finally 
        
        
                
                
        

class ResNet(nn.Module):
    def __init__(self):               # No parameters as our underlying architecture is fixed here 
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7,stride=2,padding=3)          # inchannels out channels kernel size # This layer is not a part of a block 
        
        self.batch_norm = nn.BatchNorm2d(64)               # For the Above standalone convolution layer 
        
        self.block1 = Block(64,3,64,3)          # in_channels,kernel_size=3,num_filters,num_sub_blocks
        self.block2 = Block(64,3,128,4)
        self.block3 = Block(128,3,256,6)
        self.block4 = Block(256,3,512,3)
        
        self.avg_pool = nn.AvgPool2d(7,stride=1,padding=0)           # Before the fc layer 
        
        self.fc1 = nn.Linear(512,1000)
        self.fc2 = nn.Linear(1000,5)            
        
        self.ReLu = nn.ReLU()
        
        # The 4 sub_blocks are our building blocks here 
        
        
        
    def forward(self,x):                           # Information flow through the blocks here 
        
        x1 = self.conv1(x)
        x1 = self.batch_norm(x1)
        x1 = self.ReLu(x1)
        
        x2 = self.block1(x1)
        
        x3 = self.block2(x2)
        
        x4 = self.block3(x3)
        
        x5 = self.block4(x4)
        
        x6 = self.avg_pool(x5)
        
        x6 = torch.squeeze(x6)
        
        
        x7 = self.fc1(x6)
        
        x7 = self.ReLu(x7)
        
        out = self.fc2(x7)
        return out 
        

# resnet = ResNet()
# resnet = nn.Sequential(resnet)


# resnet.apply(weights_init)


# print(resnet)

