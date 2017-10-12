import torch.nn as nn
import torch.nn.functional as F

class VQAModel(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)
    """

    def __init__(self, cnn, encoder, decoder, decode_function=F.log_softmax):
        super(VQAModel, self).__init__()
        self.cnn = cnn
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, image, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0):
        feature = self.cnn(image)
        # Initiali hidden state for encoder rnn
        h0 = feature.unsqueeze(0)
        if self.decoder.rnn_cell is nn.LSTM:
            h0 = (h0, h0)
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths, h0)
        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result
