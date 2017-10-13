import json
import logging

import torch
from torch.autograd import Variable 
from torchvision import transforms 
from PIL import Image
import nltk

from models import VQGModel, QAModel, VQAModel, TopKDecoder
from VisualGenomeQA import load_vocab, Vocabulary

class ModelWrapper(object):
    """ Wrapper class for pytorch models

    This class provides interface for pytorch models.

    Attributes:
        - model(nn.Module): pytorch model instance

    Methods:
        - init_model: initialize a model from a config file
        - predict:

    """
    def __init__(self, Model, config_path):
        self.logger = logging.getLogger(__name__)
        self.model = self.init_model(Model, config_path)

    def init_model(self, Model, config_path):

        with open(config_path) as c:
            config = json.load(c)

        # Get model config
        model_config = config[Model.__name__]
        self.logger.debug('%s config:' % (Model.__name__))
        self.logger.debug(model_config)

        # Load vocab
        self.logger.debug('Loading vocab from %s...' %(model_config['vocab_path']))
        self.vocab = load_vocab(model_config['vocab_path'])

        # Initialize model
        self.logger.debug('Initializing model...')
        model = Model(len(self.vocab),
                      model_config['max_len'],
                      model_config['hidden_size'],
                      self.vocab(self.vocab.sos),
                      self.vocab(self.vocab.eos))

        # Load model weights
        model.load_state_dict(torch.load(model_config['model_path']))


        self.beam_size = model_config['beam_size']
        model.decoder = TopKDecoder(model.decoder, self.beam_size)
        model.eval()

        if torch.cuda.is_available():
            model.cuda()

        self.logger.debug('Done')

        return model


    def decode_topk(self, *args):
        """ Given a list of inputs, output list of topk outputs for each
            input.
        """

        topk_outputs = []
        softmax_list, _, other = self.model(*args)
        topk_length = other['topk_length']
        topk_sequence = other['topk_sequence']

        for i in range(args[0].size(0)):
            outputs = []
            for k in range(self.beam_size):
                length = topk_length[i][k]
                sequence = [seq[i, k] for seq in topk_sequence]
                tgt_id_seq = [sequence[di].data[0] for di in range(length)]
                tgt_seq = [self.vocab.idx2word[tok] for tok in tgt_id_seq]
                outputs.append(' '.join(tgt_seq[:-1]))

            topk_outputs.append(outputs)

        return topk_outputs


    def predict(self, *args, **kwargs):
        raise NotImplementedError()


class VQGWrapper(ModelWrapper):
    """ Wrapper class for VQGModel"""
    def __init__(self, config_path):
        super(VQGWrapper, self).__init__(VQGModel, config_path)
        self.transform = transforms.Compose([ 
            transforms.Scale(244),
            transforms.CenterCrop(244),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def predict(self, img_paths):
        """
        Given a list of images, generate topk questions for each image

        Args:
            img_paths([str]) - list of image paths

        Output:
            topk_questions ([[str]]) - list of list of questions
        """

        images_var = []

        # Create input variable of images
        for img_path in img_paths:
            image = Image.open(img_path)
            image_tensor = Variable(self.transform(image))
            images_var.append(image_tensor)

        images_var = torch.stack(images_var)

        if torch.cuda.is_available():
            images_var = images_var.cuda()

        # Run the model
        topk_questions = self.decode_topk(images_var)

        return topk_questions


class QAWrapper(ModelWrapper):
    """ Wrapper class for QAModel"""
    def __init__(self, config_path):
        super(QAWrapper, self).__init__(QAModel, config_path)

    def predict(self, questions):
        """ Given a list of questions, generate topk answers for each question

        Args:
            answers[[str]] - list of topk answers for each question
        """
        # RNN requires input sequences to be sorted by length
        questions.sort(key = lambda s: len(s), reverse=True)

        input_lengths = [len(question) for question in questions]
        max_length = max(input_lengths)
        questions_var = torch.zeros(len(questions), max_length).long()

        # Create input variable of questions
        for i, question in enumerate(questions):
            tokens = nltk.tokenize.word_tokenize(str(question).lower())
            question = []
            question.extend([self.vocab(token) for token in tokens])
            end = len(question)

            question_tensor = torch.Tensor(question).long()
            questions_var[i, :end] = question_tensor[:end]

        questions_var = Variable(questions_var)


        if torch.cuda.is_available():
            questions_var = questions_var.cuda()

        # Run the model
        topk_answers = self.decode_topk(questions_var, input_lengths)

        return topk_answers


class VQAWrapper(ModelWrapper):
    """ Wrapper class for VQAModel"""
    def __init__(self, config_path):
        super(VQAWrapper, self).__init__(VQAModel, config_path)
        self.transform = transforms.Compose([ 
            transforms.Scale(244),
            transforms.CenterCrop(244),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def predict(self, img_questions):
        """ Given a list of image-question pairs, generate topk answers for
            each image-question pair

        Args:
            img_questions ([tuple(img_path, question)])
        """

        img_paths = [pair[0] for pair in img_questions]
        questions = [pair[1] for pair in img_questions]

        # RNN requires input sequences to be sorted by length
        questions.sort(key = lambda s: len(s), reverse=True)

        images_var = []
        input_lengths = [len(question) for question in questions]
        max_length = max(input_lengths)
        questions_var = torch.zeros(len(questions), max_length).long()

        # Create input variable of images and questions
        for img_path in img_paths:
            image = Image.open(img_path)
            image_tensor = Variable(self.transform(image))
            images_var.append(image_tensor)

        images_var = torch.stack(images_var)

        # Create input variable of questions
        for i, question in enumerate(questions):
            tokens = nltk.tokenize.word_tokenize(str(question).lower())
            question = []
            question.extend([self.vocab(token) for token in tokens])
            end = len(question)

            question_tensor = torch.Tensor(question).long()
            questions_var[i, :end] = question_tensor[:end]

        questions_var = Variable(questions_var)

        if torch.cuda.is_available():
            images_var =  images_var.cuda()
            questions_var = questions_var.cuda()

        # Run the model
        topk_answers = self.decode_topk(images_var, questions_var,
                                        input_lengths)

        return topk_answers

