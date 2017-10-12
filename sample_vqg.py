import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torch.autograd import Variable 
from torchvision import transforms 
from PIL import Image
import nltk

from VisualGenomeQA import get_loader, load_vocab, Vocabulary
from models import EncoderCNN, DecoderRNN, TopKDecoder, VQGModel

def main(args):
    # Image preprocessing
    transform = transforms.Compose([ 
        transforms.Scale(args.crop_size),  
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Load vocabulary wrapper
    vocab = load_vocab(args.vocab_path)

    # Build Models
    encoder = EncoderCNN(args.hidden_size)
    decoder = DecoderRNN(len(vocab), args.max_length, args.hidden_size,
                         sos_id=vocab(vocab.sos), eos_id=vocab(vocab.eos),
                         rnn_cell=args.rnn_cell)
    

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    vqg = VQGModel(encoder, TopKDecoder(decoder, args.beam_size))
    vqg.eval()

    # Prepare Image       
    image = Image.open(args.image)
    image_tensor = Variable(transform(image).unsqueeze(0))

    # If use gpu
    if torch.cuda.is_available():
        vqg.cuda()
        image_tensor = image_tensor.cuda()

    
    # Run model
    softmax_list, _, other = vqg(image_tensor)
    topk_length = other['topk_length'][0]
    topk_sequence = other['sequence']

    for k in range(args.beam_size):
        length = topk_length[k]
        sequence = [seq[k] for seq in topk_sequence]
        tgt_id_seq = [sequence[di][0].data[0] for di in range(length)]
        tgt_seq = [vocab.idx2word[tok] for tok in tgt_id_seq]
        print tgt_seq


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True,
                        help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str,
                        default='./weights/vqg/encoder-1-1000.pkl',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str,
                        default='./weights/vqg/decoder-1-1000.pkl',
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str,
                        default='VisualGenomeQA/data/vocab_without_answers.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for center cropping images')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--rnn_cell', type=str, default='lstm',
                        help='type of rnn cell (gru or lstm)')
    parser.add_argument('--embed_size', type=int , default=512,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    parser.add_argument('-bs', '--beam_size', type=int, default=10,
                        help='size of the beam for decoder')
    parser.add_argument('--max_length', type=int , default=20 ,
                        help='maximum sequence length')

    args = parser.parse_args()
    main(args)
