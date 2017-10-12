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
from models import TopKDecoder, VQGModel

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
    vqg = VQGModel(vocab, args.max_length, args.hidden_size,
                   rnn_cell=args.rnn_cell)

    # Load the trained model parameters
    vqg.load_state_dict(torch.load(args.model_path))

    vqg.decoder = TopKDecoder(vqg.decoder, args.beam_size)
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
    topk_length = other['topk_length']
    topk_sequence = other['topk_sequence']

    for i in range(image_tensor.size(0)):
        print 'image %d' % (i)
        for k in range(args.beam_size):
            length = topk_length[i][k]
            sequence = [seq[i, k] for seq in topk_sequence]
            tgt_id_seq = [sequence[di].data[0] for di in range(length)]
            tgt_seq = [vocab.idx2word[tok] for tok in tgt_id_seq]
            print tgt_seq


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True,
                        help='input image for generating caption')
    parser.add_argument('--model_path', type=str,
                        default='./weights/vqg-1-1000.pkl',
                        help='path for trained model')
    parser.add_argument('--vocab_path', type=str,
                        default='VisualGenomeQA/data/vocab_without_answers.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for center cropping images')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--rnn_cell', type=str, default='lstm',
                        help='type of rnn cell (gru or lstm)')
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
