import torch.nn as nn
import torch.nn.functional as F
from .EncoderCNN import EncoderCNN
from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN

class VQAModel(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)
    """

    def __init__(self, vocab_size, max_len, hidden_size,
            sos_id, eos_id,
            n_layers=1, rnn_cell='lstm', bidirectional=False,
            input_dropout_p=0, dropout_p=0, use_attention=False):

        super(VQAModel, self).__init__()
        self.encoder_cnn = EncoderCNN(hidden_size)
        self.encoder_rnn  = EncoderRNN(vocab_size, max_len, hidden_size,
                                      input_dropout_p=input_dropout_p,
                                      dropout_p=dropout_p,
                                      n_layers=n_layers,
                                      bidirectional=bidirectional,
                                      rnn_cell=rnn_cell,
                                      variable_lengths=True)

        self.decoder = DecoderRNN(vocab_size, max_len, hidden_size,
                                  sos_id=sos_id,
                                  eos_id=eos_id,
                                  n_layers=1,
                                  rnn_cell=rnn_cell,
                                  bidirectional=bidirectional,
                                  input_dropout_p=input_dropout_p,
                                  dropout_p=dropout_p,
                                  use_attention=use_attention)


    def params_to_train(self):
        params = list(self.decoder.parameters())\
                 + list(self.encoder_rnn.parameters())\
                 + list(self.encoder_cnn.cnn.fc.parameters())\
                 + list(self.encoder_cnn.bn.parameters())
        return params

    def flatten_parameters(self):
        self.encoder_rnn.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, image, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0, decode_function=F.log_softmax):
        feature = self.encoder_cnn(image)
        # Initiali hidden state for encoder rnn
        h0 = feature.unsqueeze(0)
        if self.decoder.rnn_cell is nn.LSTM:
            h0 = (h0, h0)
        encoder_outputs, encoder_hidden = self.encoder_rnn(input_variable, input_lengths, h0)
        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result
