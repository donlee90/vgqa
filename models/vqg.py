import torch.nn as nn
import torch.nn.functional as F

class VQGModel(nn.Module):
    """ Simple CNN-RNN model to generate questions from an image

    Args:
        encoder (EncoderCNN): object of EncoderCNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

    """

    def __init__(self, encoder, decoder, decode_function=F.log_softmax):
        super(VQGModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function

    def flatten_parameters(self):
        self.decoder.rnn.flatten_parameters()

    def forward(self, image, target=None,
                teacher_forcing_ratio=0):
        feature = self.encoder(image)
        # TODO: unsqueeze(num_layers * num_directions)
        feature = feature.unsqueeze(0)
        if self.decoder.rnn_cell is nn.LSTM:
            feature = (feature, feature)
        result = self.decoder(inputs=target,
                              encoder_hidden=feature,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result
