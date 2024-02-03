import os
import torch


class Flattening(torch.nn.Module):
    """Returns a flattened version of a torch tensor

    Args:
        x: the tensor to flatten

    Returns:
        the flattened tensor
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class LocalResponseNorm(torch.nn.Module):
    def __init__(self, size=5, alpha=0.0001, beta=0.75, k=1.0) -> None:
        super(LocalResponseNorm, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.local_response_norm(
            x, self.size, alpha=self.alpha, beta=self.beta, k=self.k
        )


class CNNLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cnn_kernel_size: tuple,
        cnn_stride: tuple,
        mp_kernel_size: tuple,
        mp_stride: tuple,
    ) -> None:

        torch.nn.Module.__init__(self)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_stride = cnn_stride
        self.mp_kernel_size = mp_kernel_size
        self.mp_stride = mp_stride

        self.conv_layers = torch.nn.Sequential(
            # 1st convolutional layer
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=cnn_kernel_size,
                stride=cnn_stride,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=mp_kernel_size, stride=mp_stride),
        )

    def get_output_size(self, input_size: tuple[int, int]) -> tuple[int, int]:
        cnn_output_size = (
            ((input_size[0] - self.cnn_kernel_size[0]) // self.cnn_stride[0]) + 1,
            ((input_size[1] - self.cnn_kernel_size[1]) // self.cnn_stride[1]) + 1,
        )

        return (
            ((cnn_output_size[0] - self.mp_kernel_size[0]) // self.mp_stride[0]) + 1,
            ((cnn_output_size[1] - self.mp_kernel_size[1]) // self.mp_stride[1]) + 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x)


class SaveableModel(torch.nn.Module):
    def __init__(self, name: str, *args, **kwargs) -> None:
        """A model that can be saved"""
        super().__init__(*args, **kwargs)

        self.name = name

    def save_path(self, epoch: int):
        """Returns a save path for a given model name and epoch.

        Parameters
        ----------
        name: str
            the name of the model to save
        epoch: int
            the epoch of the model to save
        """
        out_dir = os.path.join("~", "output")
        
        # create saving dir if it doesn't exist yet
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        return os.path.join(out_dir, f"{self.name}_{epoch}.pth")
    
    def save(self, epoch: int):
        """Save the model.
        
        Parameters
        ----------
        epoch: int
            the current epoch
        """
        path = self.save_path(epoch=epoch)
        torch.save(self.state_dict(), path)



class SequentialSaveableModel(SaveableModel, torch.nn.Sequential):
    
    def __init__(self, *seq: tuple[torch.nn.Module,str]):
        """Create a model from a list of layers/sub-models
        
        Parameters
        ----------
        seq: list[tuple[torch.nn.Module, str]]
            A list of layers with their names
        """
        models = [model for model, _ in seq]
        names = [name for _, name in seq]

        self.sequence = seq
        full_name = "_".join(names)


        SaveableModel.__init__(self, full_name)

        torch.nn.Sequential.__init__(self, *models)

