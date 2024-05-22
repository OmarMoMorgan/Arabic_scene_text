from torch import nn 

class CNNBlock(nn.Module):
    def __init__(self,cin,cout,stride=1,groups=1):
        '''
        I could have omitted stride info or the cin info
        '''
        super().__init__()
        self.downsample = False

        self.cnn = nn.Conv2d(cin,cout,3,padding=1,stride=stride,bias=False,groups=groups)
        self.cnn2 = nn.Conv2d(cout,cout,3,padding=1,bias=False,groups=groups)
        if stride !=1:
            self.projection  = nn.Conv2d(cin,cout,1,stride=stride,bias=False,groups=groups)
            self.downsample = True
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.BN1 = nn.BatchNorm2d(cout)
        self.BN2 = nn.BatchNorm2d(cout)
    def forward(self,x):
        out = self.act1(self.BN1(self.cnn(x)))
        if self.downsample: return self.act2(self.BN2(self.cnn2(out)) + self.projection(x))
        return self.act2(self.BN2(self.cnn2(out)) + x)

class sub_CNN(nn.Module):
    def __init__(self,
                 layersPerStage,groups=1,num_filters = [16,32,64]):
        super().__init__()
        #f1,f2,f3= (16,32,64) #the number of filters in each stage 
        f1 = num_filters[0]
        f2 = num_filters[1]
        f3 = num_filters[2]
        self.baseCNN = nn.Conv2d(3,f1,3,padding=1)
        self.base = nn.Sequential(nn.BatchNorm2d(f1),
                                  nn.ReLU())
        self.stage1= nn.Sequential(*[CNNBlock(f1,f1,groups=groups)
                                    for _ in range(layersPerStage[0])])
        layers= [CNNBlock(f1,f2,2,groups=groups)]
        layers.extend([CNNBlock(f2,f2,groups=groups) for _ in range(layersPerStage[1]-1)])
        self.stage2 = nn.Sequential(*layers)

        layers= [CNNBlock(f2,f3,2,groups=groups)]
        layers.extend([CNNBlock(f3,f3,groups=groups) for _ in range(layersPerStage[2]-1)])
        self.stage3 = nn.Sequential(*layers)

        self.head = nn.Sequential(
                            nn.AvgPool2d(2,2),
                            nn.Flatten(),
                            nn.Linear(f3*4*4,10)

        )
    def forward(self,x):
        o1 = self.base(self.baseCNN(x))
        o2 = self.stage1(o1)
        o3 = self.stage2(o2)
        o4 = self.stage3(o3)
        return self.head(o4)