# Try to add the non-local block here ::
# The non-local block is a bit complicated here :: 

# Okay so the non-local block would be inserted instead of the residual connection :: 

# Here one should note that if Wz is 0 then it serves the same purpose as the residual connection here :: 

# The process is as follows :: 

# We insert the non-local block only in a sub-block :: one can also do it in all sub-blocks it's a matter of training 
# resources available here 

# Insert into a residual connection here 

# We put the non-local block in the last sub block here :: 

# We do not change the number of filters after the 1x1 convolution here :: 

# The non-local block has multiplications and termwise additions here :: 


# Softmax computation is not that straightforward here :: 

# Followed by residual connection here that's it 
 
# Okay the size is matching here good to go here :: 

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    
# Okay we create a separate Non local sub-block class here :: 
# We do everything by default here :: Instead of the Last resnet block chunk we add the non-local block
# the first size reducing sub-block would be present followed by size invariant non-local blocks here :: 
# The non-local block would neither change the size nor the number of filters of the activation map here :: 
# We put a non local block after the last residual block here 

class non_local_sub_block(nn.Module):
    
    def __init__(self,num_filters):
        
        super().__init__()
         
        
        self.theta = nn.Conv2d(num_filters,num_filters,1,stride=1,padding=0)       # Doesn't change the spatial dimension and also num_filters is same here 
        
        self.phi = nn.Conv2d(num_filters,num_filters,1,stride=1,padding=0) 
        
        self.g = nn.Conv2d(num_filters,num_filters,1,stride=1,padding=0) 
        
        self.z = nn.Conv2d(num_filters,num_filters,1,stride=1,padding=0) 
        
        # The above 1x1 convolution layers would be needed here :: 
        self.batchnorm1 = nn.BatchNorm2d(num_filters)       # See different combinations of batchnorm here ::
#         self.batchnorm2 = nn.BatchNorm2d(num_filters)

        self.ReLu = nn.ReLU()
        
    # Input is BatchsizexHeightxWidth here :: mostly square inputs are present here :: 
    
    def forward(self,x):       # Before the Nonlocal subblock we should have a size reducing first sub block here 
        
        theta = self.theta(x)                  # BatchxHxWxnum_channels 
        
        phi = self.phi(x) 
        
        g =  self.g(x)                         # BatchsizexHxWxnum_channels here too 
        
        
        # theta phi multiplication 
        phi = phi.permute(0,3,1,2)                 # Bring channels at the front here 
        
        res = torch.matmul(theta.view(theta.size()[0],theta.size()[1]*theta.size()[2],theta.size()[-1]),phi.view(phi.size()[0],phi.size()[1],phi.size()[2]*phi.size(3)))
        res = res.view(theta.size()[0],theta.size()[1]*theta.size()[2],theta.size()[1]*theta.size()[2])
        
        # The result is in the form BatchsizexHWxHW    softmax 
        
        res = torch.nn.functional.softmax(res,dim=-1).view(theta.size()[0],theta.size()[1],theta.size()[2],theta.size()[1],theta.size()[2])
        
        
        
        # g multiplication 
        
        res = torch.matmul(res.view (theta.size()[0],theta.size()[1]*theta.size()[2],theta.size()[1]*theta.size()[2]),g.view (g.size()[0],g.size()[1]*g.size()[2],g.size()[-1]))
        res = res.view(res.size()[0],theta.size()[1],theta.size()[2],g.size()[-1])
        
        
        # Wz here :: 
        
        res = self.z(res)
        
        # Batchnorm 
        
        res = self.batchnorm1(res)
        
        # Residual 
        out = x + res
        
        # ReLU 
        
        out = self.ReLu(out)
        
        # Dimension debugging in case of error needs to be done here 
        
        return out 

# A different class for the non local block as well here :: 


class non_local_Block(nn.Module):
    
    def __init__(self,in_channels,num_sub_blocks):
        
        super().__init__()
        self.in_channels = in_channels              # in_channels = num_filters for the non local block here 1x1 convolutions 
        self.num_sub_blocks = num_sub_blocks        # How many non-local blocks do we need to add is taken here 
        
        self.sub_block_list = nn.ModuleList([])             # Initialize empty module list here 
        
        for i in range(self.num_sub_blocks):
            self.sub_block_list.append(non_local_sub_block(in_channels))        # These many non-local sub blocks would be added to the Block here 
            
            
        
        
    def forward(self,x):
        
        
        out = x                                  # Input would be traversed across all the layers here 
        
        for i in range(self.num_sub_blocks):
            
            out = self.sub_block_list[i](out)                # Pass through each of the sub blocks here :: for some number of iterations here 
            
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
        
        # Now we create a list or array of sub-blocks here :: 
        
        # We create a module list for our sub-block modules here :: 
        
        self.sub_block_list = nn.ModuleList([first_sub_block(in_channels,kernel_size,num_filters)])          # Added separately here :: 
        
        
        for i in range(self.num_iterations-1):
            self.sub_block_list.append(sub_block(num_filters,kernel_size))       # Normal sub-blocks append to the end of the list 
        
        
         
        
        
    def forward(self,x):                             # Remember the skip connections across the sub_blocks 
        
        # We will try to put everything in a modulelist here :: 
        
        out = x                                  # Input would be traversed across all the layers here 
        
        for i in range(self.num_iterations):
            
            out = self.sub_block_list[i](out)
            
        return out                                # Finally 
        

                        

class ResNet_NLBlock(nn.Module):
    def __init__(self):               # No parameters as our underlying architecture is fixed here 
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7,stride=2,padding=3)          # inchannels out channels kernel size # This layer is not a part of a block 
        
        self.batch_norm = nn.BatchNorm2d(64)               # For the Above standalone convolution layer 
        # 224 112 56 28 14 7 
        self.block1 = Block(64,3,64,3)          # in_channels,kernel_size=3,num_filters,num_sub_blocks
        self.block2 = Block(64,3,128,4)
        self.block3 = Block(128,3,256,6)
        self.block4 = Block(256,3,512,3)            # Instead of this we add a Non-local block here :: 
# We put a non local block after the last residual block here 
        self.block5 = non_local_Block(512,1)                   # Num Filters and Number of non local sub blocks here
        
        
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
        
        x5 = self.block5(x5)         # This Block Doesn't change the filter size as well as the number of channels here :: 
        
        x6 = self.avg_pool(x5)
        
        x6 = torch.squeeze(x6)
        
        
        x7 = self.fc1(x6)
        
        x7 = self.ReLu(x7)
        
        out = self.fc2(x7)
        
        return out 
        
        
        
        
# train 



