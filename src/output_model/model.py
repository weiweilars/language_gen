import re
import torch
import torch.nn as nn
from src.output_model.decoder import FPNDecoder
from src.output_model.encoder import ResNetEncoder
from torchvision.models.resnet import Bottleneck

ENCODER_PARAMS = {"out_channels": (3, 64, 256, 512, 1024, 2048),
                  "block": Bottleneck,
                  "layers": [3, 4, 6, 3],
                  "groups": 32,
                  "in_channels": 1,
                  "width_per_group": 4}

DECODER_PARAMS = {"encoder_depth": 5,
                  "pyramid_channels": 256,
                  "segmentation_channels": 128,
                  "merge_policy": "add",
                  "dropout": 0.2}

SEGHEAD_PARAMS = {"out_channels": 10,
                  "activation": nn.Identity(),
                  "upsampling": 4}

CLASSHEAD_PARMS = {"classes": 2,
                   "activation": nn.Identity()}

def rename_layers(state_dict, rename_in_layers):
    result = {}
    for key, value in state_dict.items():
        for key_r, value_r in rename_in_layers.items():
            key = re.sub(key_r, value_r, key)

        result[key] = value

    return result

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()

        activation = activation
        super().__init__(conv2d, upsampling, activation)

class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = activation
        super().__init__(pool, flatten, dropout, linear, activation)
        

class SegmentationModel(torch.nn.Module):

    def __init__(
            self,
            encoder_params = ENCODER_PARAMS,
            decoder_params = DECODER_PARAMS,
            seghead_params = SEGHEAD_PARAMS,
            clshead_params = CLASSHEAD_PARMS
    ):
        super().__init__()

        self.encoder = ResNetEncoder(**encoder_params)

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            **decoder_params)

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            kernel_size=1,
            **seghead_params)

        self.classification_head = ClassificationHead(
            in_channels=self.encoder.out_channels[-1],
            **clshead_params)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)
        labels = self.classification_head(features[-1])

        return masks, labels

class SegPredModel:
    def __init__(self,
                 path_to_model,
                 device):

        state_dict, self.hparams = self.load_model_dict(path_to_model)

        self.model = SegmentationModel()

        self.model.load_state_dict(state_dict)

        self.model = self.model.to(device).eval()

        self.device = device
        

    def load_model_dict(self, model_path):

        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        state_dict = rename_layers(state_dict, {"model.": ""})

        return state_dict, checkpoint['hyper_parameters']

    def predict(self, input):
        input = input.to(self.device)
        logits, labels = self.model(input)
        y_pred = logits.log_softmax(dim=1)
        y_pred = torch.argmax(y_pred, dim=1)

        l_pred = labels.log_softmax(dim=1)
        l_pred = torch.argmax(l_pred, dim=1)
        return y_pred.cpu().numpy()[0], l_pred.cpu().numpy()[0]