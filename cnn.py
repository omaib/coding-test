import torch.nn as nn
import torch.nn.functional as F


class SmallCNNFeature(nn.Module):
    """
    A feature extractor for small 32x32 images (e.g. CIFAR, MNIST) that outputs a feature vector of length 128.

    Args:
        num_channels (int): the number of input channels (default=3).
        kernel_size (int): the size of the convolution kernel (default=5).

    Examples::
        >>> feature_network = SmallCNNFeature(num_channels)
    """

    def __init__(self, num_channels=3, kernel_size=5):
        super(SmallCNNFeature, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64 * 2, kernel_size=kernel_size)
        self.bn3 = nn.BatchNorm2d(64 * 2)
        self.sigmoid = nn.Sigmoid()
        self._out_features = 128

    def forward(self, input_):
        x = self.bn1(self.conv1(input_))
        x = self.relu1(self.pool1(x))
        x = self.bn2(self.conv2(x))
        x = self.relu2(self.pool2(x))
        x = self.sigmoid(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x

    def output_size(self):
        return self._out_features
    
    
class SignalVAEEncoder(nn.Module):
    """
    SignalVAEEncoder encodes 1D signals into a latent representation suitable for variational autoencoders (VAE).

    This encoder uses a series of 1D convolutional layers to extract hierarchical temporal features from generic 1D signals,
    followed by fully connected layers that output the mean and log-variance vectors for the latent Gaussian distribution.
    This structure is commonly used for unsupervised or multimodal learning on time-series or sequential data.

    Args:
        input_dim (int, optional): Length of the input 1D signal (number of time points). Default is 60000.
        latent_dim (int, optional): Dimensionality of the latent space representation. Default is 256.

    Forward Input:
        x (Tensor): Input signal tensor of shape (batch_size, 1, input_dim).

    Forward Output:
        mean (Tensor): Mean vector of the latent Gaussian distribution, shape (batch_size, latent_dim).
        log_var (Tensor): Log-variance vector of the latent Gaussian, shape (batch_size, latent_dim).

    Example:
        encoder = SignalVAEEncoder(input_dim=60000, latent_dim=128)
        mean, log_var = encoder(signals)
    """

    def __init__(self, input_dim=60000, latent_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64 * (input_dim // 8), latent_dim)
        self.fc_log_var = nn.Linear(64 * (input_dim // 8), latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        mean = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mean, log_var
    
    
class ProteinCNN(nn.Module):
    """
    A protein feature extractor using Convolutional Neural Networks (CNNs).

    This class extracts features from protein sequences using a series of 1D convolutional layers.
    The input protein sequence is first embedded and then passed through multiple convolutional
    and batch normalization layers to produce a fixed-size feature vector.

    Args:
        embedding_dim (int): Dimensionality of the embedding space for protein sequences.
        num_filters (list of int): A list specifying the number of filters for each convolutional layer.
        kernel_size (list of int): A list specifying the kernel size for each convolutional layer.
        padding (bool): Whether to apply padding to the embedding layer.
    """

    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        # self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v


class LeNet(nn.Module):
    """LeNet is a customizable Convolutional Neural Network (CNN) model based on the LeNet architecture, designed for
    feature extraction from image and audio modalities.
       LeNet supports several layers of 2D convolution, followed by batch normalization, max pooling, and adaptive
       average pooling, with a configurable number of channels.
       The depth of the network (number of convolutional blocks) is adjustable with the 'additional_layers' parameter.
       An optional linear layer can be added at the end for further transformation of the output, which could be useful
       for various tasks such as classification or regression. The 'output_each_layer' option allows for returning the
       output of each layer instead of just the final output, which can be beneficial for certain tasks or for analyzing
       the intermediate representations learned by the network.
       By default, the output tensor is squeezed before being returned, removing dimensions of size one, but this can be
       configured with the 'squeeze_output' parameter.

    Args:
        input_channels (int): Input channel number.
        output_channels (int): Output channel number for block.
        additional_layers (int): Number of additional blocks for LeNet.
        output_each_layer (bool, optional): Whether to return the output of all layers. Defaults to False.
        linear (tuple, optional): Tuple of (input_dim, output_dim) for optional linear layer post-processing. Defaults to None.
        squeeze_output (bool, optional): Whether to squeeze output before returning. Defaults to True.

    Note:
        Adapted code from https://github.com/slyviacassell/_MFAS/blob/master/models/central/avmnist.py.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        additional_layers,
        output_each_layer=False,
        linear=None,
        squeeze_output=True,
    ):
        super(LeNet, self).__init__()
        self.output_each_layer = output_each_layer
        self.conv_layers = [nn.Conv2d(input_channels, output_channels, kernel_size=5, padding=2, bias=False)]
        self.batch_norms = [nn.BatchNorm2d(output_channels)]
        self.global_pools = [nn.AdaptiveAvgPool2d(1)]

        for i in range(additional_layers):
            self.conv_layers.append(
                nn.Conv2d(
                    (2**i) * output_channels, (2 ** (i + 1)) * output_channels, kernel_size=3, padding=1, bias=False
                )
            )
            self.batch_norms.append(nn.BatchNorm2d(output_channels * (2 ** (i + 1))))
            self.global_pools.append(nn.AdaptiveAvgPool2d(1))

        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.batch_norms = nn.ModuleList(self.batch_norms)
        self.global_pools = nn.ModuleList(self.global_pools)
        self.squeeze_output = squeeze_output
        self.linear = None

        if linear is not None:
            self.linear = nn.Linear(linear[0], linear[1])

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        intermediate_outputs = []
        output = x
        for i in range(len(self.conv_layers)):
            output = F.relu(self.batch_norms[i](self.conv_layers[i](output)))
            output = F.max_pool2d(output, 2)
            global_pool = self.global_pools[i](output).view(output.size(0), -1)
            intermediate_outputs.append(global_pool)

        if self.linear is not None:
            output = self.linear(output)
        intermediate_outputs.append(output)

        if self.output_each_layer:
            if self.squeeze_output:
                return [t.squeeze() for t in intermediate_outputs]
            return intermediate_outputs

        if self.squeeze_output:
            return output.squeeze()
        return output


class ImageVAEEncoder(nn.Module):
    """
    ImageVAEEncoder encodes 2D image data into a latent representation for use in a Variational Autoencoder (VAE).

    Note:
        This implementation assumes the input images are 224 x 224 pixels.
        If you use images of a different size, you must modify the architecture (e.g., adjust the linear layer input).

    This encoder consists of a stack of convolutional layers followed by fully connected layers to produce the
    mean and log-variance of the latent Gaussian distribution. It is suitable for compressing image modalities
    (such as chest X-rays) into a lower-dimensional latent space, facilitating downstream tasks like reconstruction,
    multimodal learning, or generative modelling.

    Args:
        input_channels (int, optional): Number of input channels in the image (e.g., 1 for grayscale, 3 for RGB). Default is 1.
        latent_dim (int, optional): Dimensionality of the latent space representation. Default is 256.

    Forward Input:
        x (Tensor): Input image tensor of shape (batch_size, input_channels, 224, 224).

    Forward Output:
        mean (Tensor): Mean vector of the latent Gaussian distribution, shape (batch_size, latent_dim).
        log_var (Tensor): Log-variance vector of the latent Gaussian, shape (batch_size, latent_dim).

    Example:
        encoder = ImageVAEEncoder(input_channels=1, latent_dim=128)
        mean, log_var = encoder(images)
    """

    def __init__(self, input_channels=1, latent_dim=256):
        super().__init__()
        # Convolutional layers for 224x224 input
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64 * 28 * 28, latent_dim)
        self.fc_log_var = nn.Linear(64 * 28 * 28, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass for 224 x 224 images.

        Args:
            x (Tensor): Input image tensor, shape (batch_size, input_channels, 224, 224)

        Returns:
            mean (Tensor): Latent mean, shape (batch_size, latent_dim)
            log_var (Tensor): Latent log-variance, shape (batch_size, latent_dim)
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        mean = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mean, log_var
