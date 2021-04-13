import torch
import torch.nn as nn
from torchvision.models import densenet121


class DenseNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.num_classes = num_classes
        self.base_feature = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(inplace=True),
                                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                          )

        self.dense_layer1 = nn.Sequential(nn.BatchNorm2d(64),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(64, 32 * 4, 1, bias=False),

                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                          )

        self.dense_layer2 = nn.Sequential(nn.BatchNorm2d(96),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(96, 128, 1, bias=False),

                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                          )

        self.dense_layer3 = nn.Sequential(nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 128, 1, bias=False),

                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                          )

        self.dense_layer4 = nn.Sequential(nn.BatchNorm2d(160),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(160, 128, 1, bias=False),

                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                          )

        self.dense_layer5 = nn.Sequential(nn.BatchNorm2d(192),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(192, 128, 1, bias=False),

                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                          )

        self.dense_layer6 = nn.Sequential(nn.BatchNorm2d(224),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(224, 128, 1, bias=False),

                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                          )

        self.transition1 = nn.Sequential(nn.BatchNorm2d(256),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(256, 128, 1, bias=False),
                                         nn.AvgPool2d(kernel_size=2, stride=2)
                                         )

        self.dense_layer2_1 = nn.Sequential(nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer2_2 = nn.Sequential(nn.BatchNorm2d(160),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(160, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer2_3 = nn.Sequential(nn.BatchNorm2d(192),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(192, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer2_4 = nn.Sequential(nn.BatchNorm2d(224),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(224, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer2_5 = nn.Sequential(nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(256, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer2_6 = nn.Sequential(nn.BatchNorm2d(288),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(288, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer2_7 = nn.Sequential(nn.BatchNorm2d(320),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(320, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer2_8 = nn.Sequential(nn.BatchNorm2d(352),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(352, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer2_9 = nn.Sequential(nn.BatchNorm2d(384),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(384, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer2_10 = nn.Sequential(nn.BatchNorm2d(416),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(416, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer2_11 = nn.Sequential(nn.BatchNorm2d(448),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(448, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer2_12 = nn.Sequential(nn.BatchNorm2d(480),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(480, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.transition2 = nn.Sequential(nn.BatchNorm2d(512),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(512, 256, 1, bias=False),
                                         nn.AvgPool2d(kernel_size=2, stride=2)
                                         )

        self.dense_layer3_1 = nn.Sequential(nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(256, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer3_2 = nn.Sequential(nn.BatchNorm2d(288),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(288, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer3_3 = nn.Sequential(nn.BatchNorm2d(320),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(320, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer3_4 = nn.Sequential(nn.BatchNorm2d(352),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(352, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer3_5 = nn.Sequential(nn.BatchNorm2d(384),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(384, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer3_6 = nn.Sequential(nn.BatchNorm2d(416),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(416, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer3_7 = nn.Sequential(nn.BatchNorm2d(448),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(448, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer3_8 = nn.Sequential(nn.BatchNorm2d(480),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(480, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer3_9 = nn.Sequential(nn.BatchNorm2d(512),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(512, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer3_10 = nn.Sequential(nn.BatchNorm2d(544),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(544, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer3_11 = nn.Sequential(nn.BatchNorm2d(576),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(576, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer3_12 = nn.Sequential(nn.BatchNorm2d(608),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(608, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer3_13 = nn.Sequential(nn.BatchNorm2d(640),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(640, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer3_14 = nn.Sequential(nn.BatchNorm2d(672),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(672, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer3_15 = nn.Sequential(nn.BatchNorm2d(704),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(704, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer3_16 = nn.Sequential(nn.BatchNorm2d(736),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(736, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer3_17 = nn.Sequential(nn.BatchNorm2d(768),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(768, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer3_18 = nn.Sequential(nn.BatchNorm2d(800),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(800, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer3_19 = nn.Sequential(nn.BatchNorm2d(832),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(832, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer3_20 = nn.Sequential(nn.BatchNorm2d(864),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(864, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer3_21 = nn.Sequential(nn.BatchNorm2d(896),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(896, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer3_22 = nn.Sequential(nn.BatchNorm2d(928),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(928, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer3_23 = nn.Sequential(nn.BatchNorm2d(960),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(960, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer3_24 = nn.Sequential(nn.BatchNorm2d(992),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(992, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.transition3 = nn.Sequential(nn.BatchNorm2d(1024),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(1024, 512, 1, bias=False),
                                         nn.AvgPool2d(kernel_size=2, stride=2)
                                         )

        self.dense_layer4_1 = nn.Sequential(nn.BatchNorm2d(512),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(512, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer4_2 = nn.Sequential(nn.BatchNorm2d(544),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(544, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer4_3 = nn.Sequential(nn.BatchNorm2d(576),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(576, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer4_4 = nn.Sequential(nn.BatchNorm2d(608),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(608, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer4_5 = nn.Sequential(nn.BatchNorm2d(640),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(640, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer4_6 = nn.Sequential(nn.BatchNorm2d(672),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(672, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer4_7 = nn.Sequential(nn.BatchNorm2d(704),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(704, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer4_8 = nn.Sequential(nn.BatchNorm2d(736),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(736, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer4_9 = nn.Sequential(nn.BatchNorm2d(768),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(768, 128, 1, bias=False),

                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                            )

        self.dense_layer4_10 = nn.Sequential(nn.BatchNorm2d(800),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(800, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer4_11 = nn.Sequential(nn.BatchNorm2d(832),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(832, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer4_12 = nn.Sequential(nn.BatchNorm2d(864),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(864, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer4_13 = nn.Sequential(nn.BatchNorm2d(896),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(896, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer4_14 = nn.Sequential(nn.BatchNorm2d(928),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(928, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer4_15 = nn.Sequential(nn.BatchNorm2d(960),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(960, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )

        self.dense_layer4_16 = nn.Sequential(nn.BatchNorm2d(992),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(992, 128, 1, bias=False),

                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                             )
        self.norm = nn.BatchNorm2d(1024)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, self.num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        print("num_params : ", self.count_parameters())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        # 6
        x = self.base_feature(x)
        x1 = self.dense_layer1(torch.cat([x], 1))
        x2 = self.dense_layer2(torch.cat([x, x1], 1))
        x3 = self.dense_layer3(torch.cat([x, x1, x2], 1))
        x4 = self.dense_layer4(torch.cat([x, x1, x2, x3], 1))
        x5 = self.dense_layer5(torch.cat([x, x1, x2, x3, x4], 1))
        x6 = self.dense_layer6(torch.cat([x, x1, x2, x3, x4, x5], 1))
        x = self.transition1(torch.cat([x, x1, x2, x3, x4, x5, x6], 1))

        # 12
        x1 = self.dense_layer2_1(torch.cat([x], 1))
        x2 = self.dense_layer2_2(torch.cat([x, x1], 1))
        x3 = self.dense_layer2_3(torch.cat([x, x1, x2], 1))
        x4 = self.dense_layer2_4(torch.cat([x, x1, x2, x3], 1))
        x5 = self.dense_layer2_5(torch.cat([x, x1, x2, x3, x4], 1))
        x6 = self.dense_layer2_6(torch.cat([x, x1, x2, x3, x4, x5], 1))
        x7 = self.dense_layer2_7(torch.cat([x, x1, x2, x3, x4, x5, x6], 1))
        x8 = self.dense_layer2_8(torch.cat([x, x1, x2, x3, x4, x5, x6, x7], 1))
        x9 = self.dense_layer2_9(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8], 1))
        x10 = self.dense_layer2_10(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9], 1))
        x11 = self.dense_layer2_11(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], 1))
        x12 = self.dense_layer2_12(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], 1))
        x = self.transition2(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], 1))

        # 24
        x1 = self.dense_layer3_1(torch.cat([x], 1))
        x2 = self.dense_layer3_2(torch.cat([x, x1], 1))
        x3 = self.dense_layer3_3(torch.cat([x, x1, x2], 1))
        x4 = self.dense_layer3_4(torch.cat([x, x1, x2, x3], 1))
        x5 = self.dense_layer3_5(torch.cat([x, x1, x2, x3, x4], 1))
        x6 = self.dense_layer3_6(torch.cat([x, x1, x2, x3, x4, x5], 1))
        x7 = self.dense_layer3_7(torch.cat([x, x1, x2, x3, x4, x5, x6], 1))
        x8 = self.dense_layer3_8(torch.cat([x, x1, x2, x3, x4, x5, x6, x7], 1))
        x9 = self.dense_layer3_9(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8], 1))
        x10 = self.dense_layer3_10(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9], 1))
        x11 = self.dense_layer3_11(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], 1))
        x12 = self.dense_layer3_12(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], 1))
        x13 = self.dense_layer3_13(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], 1))
        x14 = self.dense_layer3_14(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13], 1))
        x15 = self.dense_layer3_15(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14], 1))
        x16 = self.dense_layer3_16(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15], 1))
        x17 = self.dense_layer3_17(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16], 1))
        x18 = self.dense_layer3_18(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17], 1))
        x19 = self.dense_layer3_19(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18], 1))
        x20 = self.dense_layer3_20(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19], 1))
        x21 = self.dense_layer3_21(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20], 1))
        x22 = self.dense_layer3_22(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21], 1))
        x23 = self.dense_layer3_23(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22], 1))
        x24 = self.dense_layer3_24(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23], 1))
        x = self.transition3(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24], 1))

        x1 = self.dense_layer4_1(torch.cat([x], 1))
        x2 = self.dense_layer4_2(torch.cat([x, x1], 1))
        x3 = self.dense_layer4_3(torch.cat([x, x1, x2], 1))
        x4 = self.dense_layer4_4(torch.cat([x, x1, x2, x3], 1))
        x5 = self.dense_layer4_5(torch.cat([x, x1, x2, x3, x4], 1))
        x6 = self.dense_layer4_6(torch.cat([x, x1, x2, x3, x4, x5], 1))
        x7 = self.dense_layer4_7(torch.cat([x, x1, x2, x3, x4, x5, x6], 1))
        x8 = self.dense_layer4_8(torch.cat([x, x1, x2, x3, x4, x5, x6, x7], 1))
        x9 = self.dense_layer4_9(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8], 1))
        x10 = self.dense_layer4_10(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9], 1))
        x11 = self.dense_layer4_11(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], 1))
        x12 = self.dense_layer4_12(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], 1))
        x13 = self.dense_layer4_13(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], 1))
        x14 = self.dense_layer4_14(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13], 1))
        x15 = self.dense_layer4_15(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14], 1))
        x16 = self.dense_layer4_16(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15], 1))
        x = self.norm(torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16], 1))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # 16
        return x


if __name__ == '__main__':
    import numpy as np

    img = torch.randn([5, 3, 224, 224]).cuda()
    model = DenseNet().cuda()

    model = densenet121()
    # print(model)
    # parameter 구하는 부분
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)