import torch
import numpy as np
import torchvision


def convert_model():
    model = torchvision.models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=2, bias=True)
    model.load_state_dict(torch.load('models/best_model.pt'))
    model.eval()

    # create a dummy input tensor
    dummy_input = torch.randn(1, 3, 128, 128, requires_grad=True)
    
    # convert
    torch.onnx.export(
        model,
        dummy_input,
        'models/best_model.onnx',
        export_params=True,
        opset_version=10,
        input_names = ['modelInput'],
        output_names = ['modelOutput'],
        dynamic_axes={'modelInput': {0: 'batch_size'},
                        'modelOutput': {0: 'batch_size'}}
    )

if __name__ == '__main__':
    convert_model()
    print('Model has been converted to ONNX')