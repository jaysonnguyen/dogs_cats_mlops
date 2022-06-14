import torch
import torchvision

def mobilenet_v2(num_classes):
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes, bias=True)
    return model