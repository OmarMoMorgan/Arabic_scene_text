from torch import nn 

class ResBlock(nn.Module):
    def __init__(self,cin,cout,stride=1):
        '''
        I could have omitted stride info or the cin info
        '''
        super().__init__()
        self.downsample = False

        self.cnn = nn.Conv2d(cin,cout,3,padding=1,stride=stride,bias=False)
        self.cnn2 = nn.Conv2d(cout,cout,3,padding=1,bias=False)
        if stride !=1:
            self.projection  = nn.Conv2d(cin,cout,1,stride=stride,bias=False)
            self.downsample = True
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.BN1 = nn.BatchNorm2d(cout)
        self.BN2 = nn.BatchNorm2d(cout)
    def forward(self,x):
        out = self.act1(self.BN1(self.cnn(x)))
        if self.downsample: return self.act2(self.BN2(self.cnn2(out)) + self.projection(x))
        return self.act2(self.BN2(self.cnn2(out)) + x)

class ResNetA(nn.Module):
    def __init__(self,
                 layersPerStage):
        super().__init__()
        f1,f2,f3= (16,32,64)
        self.baseCNN = nn.Conv2d(3,f1,3,padding=1)
        self.base = nn.Sequential(nn.BatchNorm2d(f1),
                                  nn.ReLU())
        self.stage1= nn.Sequential(*[ResBlock(f1,f1)
                                    for _ in range(layersPerStage[0])])
        layers= [ResBlock(f1,f2,2)]
        layers.extend([ResBlock(f2,f2) for _ in range(layersPerStage[1]-1)])
        self.stage2 = nn.Sequential(*layers)

        layers= [ResBlock(f2,f3,2)]
        layers.extend([ResBlock(f3,f3) for _ in range(layersPerStage[2]-1)])
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