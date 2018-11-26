import torch.nn as nn
import torch.nn.functional as F
import torch

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
softmax = torch.nn.Softmax2d()

def to_binary(tensor):
    tensor[tensor > 0.5] = 1
    tensor[tensor <= 0.5] = 0
    return tensor

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def unpack_model(*model):
    return model

##############################
#           RESNET
##############################

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]


        self.conv_block = nn.Sequential(conv_block[0],conv_block[1],conv_block[2],conv_block[3],conv_block[4],conv_block[5],conv_block[6])

    def forward(self, x):
        return x + self.conv_block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, res_blocks=9):
        super(GeneratorResNet, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(in_channels, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, out_channels, 7),
                    nn.Sigmoid() ]


        self.model = nn.Sequential(model[0],model[1],model[2],model[3],model[4],model[5],model[6],model[7],model[8],model[9],model[10],model[11],model[12],model[13],model[14],model[15],model[16],model[17],model[18],model[19],model[20],model[21],model[22],model[23],model[24])

    def forward(self, x):
        output = self.model(x)
        #output size Batch_size * channels * height * weight
        #output = output[:,:,:,0:88]
        #_,output = torch.max(output,1)
        #output = torch.unsqueeze(output,1).type(Tensor)
        #result_softmax = softmax(output)
        #size is Batch_size * height * weight
        # result is the value of max value
        # index is the index of max value 
        '''
        batch_size = x.size()[0]
        result_2d = result
        result = result.type(Tensor)
        result,index_2 = torch.max(result,2)
        #size is Batch_size * height
        #result = torch.squeeze(result)
        one_hot_result = torch.zeros(batch_size,100,88)
        for k in range(batch_size):
            for i in range(100):
                one_hot_result[k][i][index_2[k][i]] = index_1[k][i][index_2[k][i]]
        one_hot_result = torch.unsqueeze(one_hot_result,1)
        one_hot_result = one_hot_result.type(Tensor)   
        
        return one_hot_result,result_softmax
        '''
        
        #index_1 = index_1.type(Tensor)
        return output
        

##############################
#        Discriminator
##############################
def discriminator_block(in_filters, out_filters, normalize=True):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
    if normalize:
        layers.append(nn.InstanceNorm2d(out_filters))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()



        self.dis_block1 = discriminator_block(in_channels, 64, normalize=False)
        self.dis_block2 = discriminator_block(64, 128)
        self.dis_block3 = discriminator_block(128, 256)
        self.dis_block4 = discriminator_block(256, 512)


        self.model = nn.Sequential(
            self.dis_block1[0],
            self.dis_block1[1],
            self.dis_block2[0],
            self.dis_block2[1],
            self.dis_block3[0],
            self.dis_block3[1],
            self.dis_block4[0],
            self.dis_block4[1],
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
