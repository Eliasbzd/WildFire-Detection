import torch

from classifiers.nn_utils import Flattening, LocalResponseNorm, CNNLayer

class CNNSimple(torch.nn.Module):
    def __init__(self, input_size: tuple[int, int]) -> tuple[int, int]:
        super(CNNSimple,self).__init__()

        self.conv_layer1 = CNNLayer(
            in_channels=1,
            out_channels=64,
            cnn_kernel_size=(7, 7),
            cnn_stride=(1, 1),
            mp_kernel_size=(3, 3),
            mp_stride=(2, 2),
        )

        self.conv_layer2 = CNNLayer(
            in_channels=64,
            out_channels=128,
            cnn_kernel_size=(5, 5),
            cnn_stride=(1, 1),
            mp_kernel_size=(3, 3),
            mp_stride=(2, 2),
        )

        self.conv_layer3 = CNNLayer(
            in_channels=128,
            out_channels=256,
            cnn_kernel_size=(3, 3),
            cnn_stride=(1, 1),
            mp_kernel_size=(3, 3),
            mp_stride=(2, 2),
        )

        """self.conv_layer4 = CNNLayer(
            in_channels=256,
            out_channels=384,
            cnn_kernel_size=(3, 3),
            cnn_stride=(1, 1),
            mp_kernel_size=(3, 3),
            mp_stride=(2, 2),
        )
        """
        """
        self.conv_layer5 = CNNLayer(
            in_channels=384,
            out_channels=256,
            cnn_kernel_size=(3, 3),
            cnn_stride=(1, 1),
            mp_kernel_size=(3, 3),
            mp_stride=(2, 2),
        )
        """
        input_size = self.conv_layer1.get_output_size(input_size=input_size)
        input_size = self.conv_layer2.get_output_size(input_size=input_size)
        #input_size = self.conv_layer3.get_output_size(input_size=input_size)
        #input_size = self.conv_layer4.get_output_size(input_size=input_size)
        #input_size = self.conv_layer5.get_output_size(input_size=input_size)

        self.flatten = Flattening()

        self.linear_layers = torch.nn.Sequential(
            # 1st linear layer
            torch.nn.LazyLinear(out_features=1000),
            torch.nn.ReLU(),
            # 2nd linear layer
            torch.nn.Linear(in_features=1000, out_features=50),
            # torch.nn.ReLU(),
        )

        self.norm = LocalResponseNorm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method for the CNN model from Bardou.
        Computes and returns prediction of the model for the data x.

        Parameters
        ----------
        x : torch.Tensor
            -- Input data tensor
        """
        x = self.conv_layer1(x)
        x = self.norm(x)
        x = self.conv_layer2(x)
        x = self.norm(x)
        #x = self.conv_layer3(x)
        #x = self.conv_layer4(x)
        #x = self.conv_layer5(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        return x
