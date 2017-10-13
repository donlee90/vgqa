import torch.nn as nn
import torch.nn.functional as F
from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN

class QAModel(nn.Module):
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

        super(QAModel, self).__init__()

        self.encoder = EncoderRNN(vocab_size, max_len, hidden_size,
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



    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0, decode_function=F.log_softmax):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)
        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result
