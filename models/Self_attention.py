# Here we try to do something new :: Incorporate an attention mechanism in the architecture :: 
# In the standalone attention paper :: the author proposes 2 blocks :: one is an attention stem block other is a simple 
# attention block :: The attention stem block is used to learn the coarse features in the earlier blocks 
# The later layers have attention blocks with spatial relative positional embedding 



# # We just add one fully attentional block in the later layers replacing the chunk of the convolutional block 
# # completely here :: We create a different class for the standalone attention block here :: 


# # A very key point which is similar to the multiple activation maps used in convolution layers 
# # Is that they use multi-head attention :: 

# # So for example we can use N = din here where din is the number of output channels from the previous convolution layer 

# # Then we learn the N attention maps and then generate the output and then concatenate :: This is very important 

# # We should also try the pretrained resnet model here to determine whether we are training with less samples at our disposal here


# Now the standalone self attention block here :: 
# Multihead self attention N = din = dout here :: channels are kept the same here :: 
# Kernel size is by default equal to 3 here :: 

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

        



class SA_sub_block(nn.Module):
    
    def __init__(self,in_channels,kernel_size):              # in_channels = num_filters = out_channels here :: 
        super().__init__()
        
        self.in_channels = in_channels 
        self.kernel_size = kernel_size 
        
        self.query_conv = nn.Conv2d(self.in_channels,self.in_channels,1,stride=1,padding=0)
        
        self.key_conv = nn.Conv2d(self.in_channels,self.in_channels,1,stride=1,padding=0)
        
        self.value_conv = nn.Conv2d(self.in_channels,self.in_channels,1,stride=1,padding=0)
        
        
        # Now the relative distance embeddings here :: 
        
        self.rel_h = nn.Parameter(torch.randn(self.in_channels // 2, 1, 1, 3, 3), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(self.in_channels // 2, 1, 1, 3, 3), requires_grad=True)
        
        self.N = self.in_channels                  # N = din = dout here 
        
        self.ReLu = nn.ReLU()
        
        self.batchnorm = nn.BatchNorm2d(self.in_channels)
        
    def forward(self,x):          
        
        batch, channels, height, width = x.size()        # This is useful information here 
        
        # Take an input activation map and get the query map :: doesn't require padding here 
        
        x_query = self.query_conv(x)
        
        
        # Do padding here :: Then compute the value and the key maps 
        
        padding = self.kernel_size//2  

        x_padded = F.pad(x, [padding, padding, padding, padding])          # From all sides here 
        
        x_key = self.key_conv(x_padded)
        
        x_value = self.value_conv(x_padded)
        
        # Extract the kernel size windows across the the image here and get the key and the value kernels 
        
        x_key_windows = x_key.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)   # unfold across dimension 2 and 3 here
        
        x_value_windows = x_value.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        
        # Padding would ensure that the first 4 dimensions of the key and query match here :: 
        
        
        # Add the relative distance embedding and take the Product with the query map 
        rel = torch.cat((self.rel_h,self.rel_w),dim=0)            # Concatenate along the channels dimension here 
        
        # Channelsx1,1,3,3
        
        x_key_rel_embedded = x_key_windows + rel                   # Relative embeddings added here 
        
        # Query is BxCxHxW not padded here :: x_key_rel is BxCxHxWxKxK 
        
        x_query = x_query.view(batch,channels,height,width)
        x_query = x_query[:,:,:,:,None,None]          # For the last two dimensions 
        x_key_rel_embedded = x_key_rel_embedded.view(batch,channels,height,width,self.kernel_size,self.kernel_size)
        
#         print(x_query.size())
#         print(x_key_rel_embedded.size())
        
        similarity = (x_query*x_key_rel_embedded).reshape(batch,channels,height,width,self.kernel_size*self.kernel_size)
        
        # Do softmax along the neighbourhood dimension here ::
        similarity = F.softmax(similarity,dim=-1).reshape(batch,channels,height,width,self.kernel_size,self.kernel_size)
        
        # Do weighted sum to get the final output here :: 
        
        out = similarity*x_value_windows           # Summation is left here 
        
#         print(out.size())
#         print(out.size())

        out = torch.sum(out,dim=[4,5])               # Last 2 dimensions sum here ::   
#         print(x.size())
        
        # Try batch normalization and the resnet residual connection here :: 
        
        out = self.batchnorm(out)
        
        out = out + x
        
        out = self.ReLu(out)
        
        return out 


class SABlock(nn.Module):
    
    def __init__(self,in_channels,kernel_size,num_sub_blocks):              # in_channels = num_filters = out_channels here :: 
        super().__init__()
        
        self.in_channels = in_channels 
        self.kernel_size = kernel_size
        self.num_sub_blocks = num_sub_blocks           # Number of the self attention blocks here 
        
        # Add a module-list for the sub_blocks here :: 
        
        self.sub_block_list = nn.ModuleList([])          # Added separately here :: 
        
        
        for i in range(self.num_sub_blocks):
            self.sub_block_list.append(SA_sub_block(self.in_channels,self.kernel_size))
            
            
    def forward(self,x):                  # Pass through all the sub blocks here :: 

        out = x 

        for i in range(self.num_sub_blocks): 
            out = self.sub_block_list[i](out)               # Pass through all the sub blocks here 


        return out 
    
    
class SA_ResNet(nn.Module):
    def __init__(self):               # No parameters as our underlying architecture is fixed here 
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7,stride=2,padding=3)          # inchannels out channels kernel size # This layer is not a part of a block 
        
        self.batch_norm = nn.BatchNorm2d(64)               # For the Above standalone convolution layer 
        
        self.block1 = Block(64,3,64,3)          # in_channels,kernel_size=3,num_filters,num_sub_blocks
        self.block2 = Block(64,3,128,4)
        self.block3 = Block(128,3,256,6)
        self.block4 = SABlock(256,3,3)              # Replacing ResNet block here 
        
        self.avg_pool = nn.AvgPool2d(14,stride=1,padding=0)           # Before the fc layer 
        
        self.fc1 = nn.Linear(256,1000)
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
        
        
        
        
# train 

# sa_resnet = SA_ResNet()
# sa_resnet = nn.Sequential(sa_resnet)
# # resnet.to(device)



# sa_resnet.apply(weights_init) 

# print(sa_resnet)


# # Okay now let's try to verify and train this model here :: 

