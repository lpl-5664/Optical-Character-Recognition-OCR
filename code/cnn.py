import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Layer, BatchNormalization, Conv2D, 
                                     Dropout, ReLU, ZeroPadding2D)

class RepVGGBlock(Layer):
    """
    A class to define the behaviour of a block in RepVGG model
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, deploy=False):
        """
        Arguments:
            in_channels: the number of channels in the input for the block
            out_channels: the number of channels in the output for the block
            kernel_size: the size of the convolutional filter
            stride: stride while computing with the convolutional filter
            deploy: a boolean, indicating the model to behave in train mode or inference mode
        """
        super(RepVGGBlock, self).__init__()
        
        self.deploy = deploy
        self.in_channels = in_channels
        
        # Padding is set to 1
        padding = 1
        # Get padding for the convolution 1x1 branch
        padding_11 = padding - kernel_size//2
        
        value = np.zeros((3, 3, int(self.in_channels), int(self.in_channels)), dtype=np.float32)
        for i in range(int(self.in_channels)):
            value[1, 1, i % int(self.in_channels), i] = 1
        self.id_tensor = tf.convert_to_tensor(value, dtype=np.float32)
        
        # Create architecture for inference mode (1 branch only)
        if self.deploy:
            self.reparam = Sequential(
                [ZeroPadding2D(padding=padding, name='pad'),
                 Conv2D(out_channels, kernel_size, strides=stride, padding='valid', 
                        dilation_rate=1, name='conv')
                ])
        # Create architecture for training mode (3 branches)
        else:
            # Main branch with 3x3 convolution
            self.conv_bn3x3 = Sequential(
                [ZeroPadding2D(padding=padding, name='pad'),
                 Conv2D(out_channels, kernel_size, strides=stride, padding='valid', 
                        dilation_rate=1, use_bias=False, name='conv'),
                 BatchNormalization(name='bn')
                ])
            
            # Sub branch with 1x1 convolution
            self.conv_bn1x1 = Sequential(
                [ZeroPadding2D(padding=padding_11, name='pad'),
                 Conv2D(out_channels, kernel_size=1, strides=stride, padding='valid', 
                        dilation_rate=1, use_bias=False, name='conv'),
                 BatchNormalization(name='bn')
                ])
            
            # Identity branch 
            self.identity = BatchNormalization(name='identity') if (stride==1 and 
                                                                in_channels == out_channels) else None
        
        self.activation = ReLU()
    
    def call(self, inputs):
        """
        Arguments:
            inputs: inputs passed into the block
        Return:
            output of the block after going through the branch(es)
        """
        # Behaviour when model is inferencing
        if hasattr(self, "reparam"):
            inference_branch = self.reparam(inputs)
            return self.activation(inference_branch)
        
        # Initialize the branches for training
        train_branch = self.conv_bn3x3(inputs)
        sub_branch = self.conv_bn1x1(inputs)
        if self.identity is None:
            id_branch = 0
        else: 
            id_branch = self.identity(inputs)
        
        return self.activation(train_branch+sub_branch+id_branch)
    
    def convert_kernel_bias(self):
        """
        A function to get weights (kernel & bias) of the branches in 
        training mode to pass to the inference mode
        """
        kernel3x3, bias3x3 = self.get_kernel_bias(self.conv_bn3x3) 
        kernel1x1, bias1x1 = self.get_kernel_bias(self.conv_bn1x1)
        kernelid, biasid = self.get_kernel_bias(self.identity)
        
        # Return the kernel & bias for inference by adding kernels 
        # to kernels and bias to bias
        return (kernel3x3+self.pad_1x1_to_3x3(kernel1x1)+kernelid,
               bias3x3+bias1x1+biasid)
    
    def pad_1x1_to_3x3(self, kernel1x1):
        """
        A function to pad the kernel of 1x1 to 3x3
        """
        if kernel1x1 is None:
            return 0
        else:
            return tf.pad(
                kernel1x1, tf.constant([[1, 1], [1, 1], [0, 0], [0, 0]])
            )
        
    def get_kernel_bias(self, branch):
        """
        A function to get the kernel & bias from a branch
        """
        if branch is None:
            return 0, 0
        
        # The case where the branch is Sequential 
        if isinstance(branch, tf.keras.Sequential):
            # Get the kernel
            kernel = branch.get_layer("conv").weights[0]
            # Get the necessary parameters
            running_mean = branch.get_layer("bn").moving_mean
            running_var = branch.get_layer("bn").moving_variance
            gamma = branch.get_layer("bn").gamma
            beta = branch.get_layer("bn").beta
            eps = branch.get_layer("bn").epsilon
        
        # The case of Identity branch
        else:
            assert isinstance(branch, tf.keras.layers.BatchNormalization)
            
            # Get kernel
            kernel = self.id_tensor
            # Get the necessary parameters
            running_mean = branch.moving_mean
            running_var = branch.moving_variance
            gamma = branch.gamma
            beta = branch.beta
            eps = branch.epsilon
        std = tf.sqrt(running_var + eps)
        t = gamma / std
        
        # Calculate the kernel and bias of the branch
        target_kernel = kernel*t
        target_bias = beta-running_mean*gamma/std
        return target_kernel, target_bias
    
    def convert_inference(self):
        """
        A function to convert the behaviour 
        to inference mode
        """
        # Get the converted kernel & bias
        kernel, bias = self.convert_kernel_bias()
        return kernel, bias
    
        
class RepVGG(Layer):
    """
    A class to implement the RepVGG
    """
    def __init__(self, num_blocks, width_multipliers, dropout=0.5, deploy=False):
        """
        Arguments:
            num_blocks: a list representing the number of blocks in each stage
            width_multipliers: the coefficients to manipulate the channels in each stage
            dropout: default=0.5
            deploy: a boolean depicting training or inference mode
        """
        super(RepVGG, self).__init__()
        
        self.deploy = deploy
        
        # Define the number of output channels of the beginning stage based on width_multipliers
        self.in_planes = min(64, 64*width_multipliers[0])
        assert len(width_multipliers) == 4
        
        # Initialize the beginning stage containing only 1 block
        self.stage0  = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, deploy=deploy)
        
        # Initialize the stages with the number of blocks indicated in num_blocks
        self.stage1  = self.make_stage(int(64*width_multipliers[0]), num_blocks[0], stride=2) 
        self.stage2  = self.make_stage(int(96*width_multipliers[1]), num_blocks[1], stride=2)
        self.stage3  = self.make_stage(int(128*width_multipliers[2]), num_blocks[2], stride=2)
        self.stage4  = self.make_stage(int(256*width_multipliers[3]), num_blocks[3], stride=2)
        #self.gap     = AveragePooling2D(pool_size=(3, 3))
        
        # Initialize the dropout layer for train
        self.dropout = Dropout(dropout)
        
    def make_stage(self, out_channels, num_blocks, stride):
        """
        Arguments:
            out_channels: the number of output channels of a stage
            num_blocks: an integer indicating the number of blocks there are in this stage
            stride: the stride for when computing the convolution
        """
        # Get the stride list for each block in the stage
        # The stride for the first block is defined by user through the argument "stride" (usually 2)
        # The strides for the rest of the blocks are default to 1
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        # Create a new block for each stride created in the list
        for stride in strides:
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=out_channels, 
                                      kernel_size=3, stride=stride, deploy=self.deploy))
            # Update the next block's input channels (= last block's output channels)
            self.in_planes = out_channels
        return Sequential(blocks)
    
    def call(self, inputs):
        # Behaviour when in inference mode
        if self.deploy:
            output = self.stage0(inputs)
            output = self.stage1(output)
            output = self.stage2(output)
            output = self.stage3(output)
            output = self.stage4(output)
            #output = self.gap(output)
            
        # Behaviour in training mode (add dropout)
        else:
            output = self.stage0(inputs)
            output = self.stage1(output)
            output = self.stage2(output)
            output = self.stage3(output)
            output = self.stage4(output)
            #output = self.gap(output)
            output = self.dropout(output)

        # reshape the output to 3-dimensional by multiplying width*height to be fed in an encoder 
        output = tf.reshape(output, [-1,
                                     output.get_shape()[1]*output.get_shape()[2],
                                     output.get_shape()[3]])
        return output