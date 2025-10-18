import torch

from cnn import (
    ImageVAEEncoder,
    LeNet,
    ProteinCNN,
    SignalVAEEncoder,
    SmallCNNFeature,
)


def test_small_cnn_feature_produces_flat_features():
    model = SmallCNNFeature(num_channels=3, kernel_size=5).eval()
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 32, 32)
    with torch.no_grad():
        output = model(input_tensor)

    assert output.shape == (batch_size, model.output_size())
    assert output.shape[1] == 128


def test_signal_vae_encoder_shapes_match_latent_dim():
    latent_dim = 10
    model = SignalVAEEncoder(input_dim=32, latent_dim=latent_dim).eval()
    input_tensor = torch.randn(3, 1, 32)
    with torch.no_grad():
        mean, log_var = model(input_tensor)

    assert mean.shape == (3, latent_dim)
    assert log_var.shape == (3, latent_dim)


def test_protein_cnn_returns_sequence_feature_map():
    model = ProteinCNN(
        embedding_dim=8,
        num_filters=[16, 32, 64],
        kernel_size=[3, 3, 3],
        padding=True,
    ).eval()
    batch_size, sequence_length = 2, 10
    token_ids = torch.randint(0, 26, (batch_size, sequence_length))

    with torch.no_grad():
        features = model(token_ids)

    assert features.shape == (batch_size, sequence_length - 6, 64)
    assert features.dtype == torch.float32


def test_lenet_output_shape_without_squeeze():
    model = LeNet(
        input_channels=1,
        output_channels=4,
        additional_layers=1,
        output_each_layer=False,
        squeeze_output=False,
    ).eval()
    input_tensor = torch.randn(2, 1, 32, 32)

    with torch.no_grad():
        output = model(input_tensor)

    assert output.shape == (2, 8, 8, 8)


def test_lenet_returns_all_intermediate_outputs_when_requested():
    additional_layers = 2
    model = LeNet(
        input_channels=1,
        output_channels=4,
        additional_layers=additional_layers,
        output_each_layer=True,
        squeeze_output=True,
    ).eval()
    input_tensor = torch.randn(2, 1, 32, 32)

    with torch.no_grad():
        outputs = model(input_tensor)

    expected_conv_blocks = 1 + additional_layers
    assert len(outputs) == expected_conv_blocks + 1
    # Global pooled features keep batch dimension and channel count.
    assert outputs[0].shape == (2, 4)
    assert outputs[1].shape == (2, 8)
    assert outputs[2].shape == (2, 16)
    assert outputs[-1].shape == (2, 16, 4, 4)


def test_image_vae_encoder_output_shapes():
    latent_dim = 32
    model = ImageVAEEncoder(input_channels=1, latent_dim=latent_dim).eval()
    input_tensor = torch.randn(2, 1, 224, 224)

    with torch.no_grad():
        mean, log_var = model(input_tensor)

    assert mean.shape == (2, latent_dim)
    assert log_var.shape == (2, latent_dim)
