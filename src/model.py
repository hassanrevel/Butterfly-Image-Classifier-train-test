import torchvision.models as models
import torch.nn as nn


def efficientmodel(pretrained, fine_tune, num_classes=10):
    if pretrained:
        print('[INFO]: Loading pretrained pre-trained weights')
    else:
        print('[INFO]: Not loading pretrained weights')

    model = models.efficientnet_b0(pretrained=pretrained)

    if fine_tune:
        print('[INFO]: Fine tuning all layers .......')
        for params in model.parameters():
            params.requires_grad = True

    elif not fine_tune:
        print('[INFO]: Freezing all hidden layers.....')
        for params in model.parameters():
            params.requires_grad = False

    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, out_features=num_classes, bias=True)

    return model
