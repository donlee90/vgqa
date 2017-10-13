import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle

from VisualGenomeQA import get_loader, load_vocab, Vocabulary
from models import QAModel
from loss import NLLLoss

from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

import logging
LOG_FORMAT='%(name)-12s %(levelname)-8s %(message)s'

def process_lengths(inputs):
    max_length = inputs.size(1)
    lengths = list(max_length - inputs.data.eq(0).sum(1).squeeze())
    return lengths

def main(args):
    # Config logging
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logger = logging.getLogger()

    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Load vocabulary wrapper.
    vocab = load_vocab(args.vocab_path)
    
    # Build data loader
    logger.info("Building data loader...")
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 
    logger.info("Done")

    # Build the models
    logger.info("Building image captioning models...")
    qa = QAModel(len(vocab), args.max_length, args.hidden_size,
                 vocab(vocab.sos), vocab(vocab.eos),
                 rnn_cell=args.rnn_cell)
    logger.info("Done")
    
    if torch.cuda.is_available():
        qa.cuda()

    # Loss and Optimizer
    weight = torch.ones(len(vocab))
    pad = vocab(vocab.pad)  # Set loss weight for 'pad' symbol to 0
    loss = NLLLoss(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    # Parameters to train
    params = list(qa.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the Models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, questions, answers) in enumerate(data_loader):
            
            # Set mini-batch dataset
            images = Variable(images)
            questions = Variable(questions)
            answers = Variable(answers)
            if torch.cuda.is_available():
                images = images.cuda()
                questions = questions.cuda()
                answers = answers.cuda()

            lengths = process_lengths(questions)
            
            # Forward, Backward and Optimize
            qa.zero_grad()
            outputs, hiddens, other = qa(questions, lengths, answers,
                                         teacher_forcing_ratio=1.0)
            # Get loss
            loss.reset()
            for step, step_output in enumerate(outputs):
                batch_size = answers.size(0)
                loss.eval_batch(step_output.contiguous().view(batch_size, -1),
                                answers[:, step + 1])
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                logger.info('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      %(epoch, args.num_epochs, i, total_step, loss.get_loss())) 
                
            # Save the models
            if (i+1) % args.save_step == 0:
                torch.save(qa.state_dict(), 
                           os.path.join(args.model_path, 
                                        'qa-%d-%d.pkl' %(epoch+1, i+1)))
                
if __name__ == '__main__':
    ROOT_DIR = '/home/donlee/QBot/qbot/cnnlstm/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./weights/' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 ,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='VisualGenomeQA/data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default=ROOT_DIR+'data/resized' ,
                        help='directory for resized images')
    parser.add_argument('--caption_path', type=str,
                        default=ROOT_DIR+'data/question_answers.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000,
                        help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--rnn_cell', type=str, default='lstm',
                        help='type of rnn cell (gru or lstm)')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    parser.add_argument('--max_length', type=int , default=20 ,
                        help='maximum sequence length')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
