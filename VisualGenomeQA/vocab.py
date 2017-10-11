import nltk
import pickle
import argparse
from collections import Counter
import json

# Configure logging
import logging
logger = logging.getLogger(__name__)

# Defalt vocab path
import os
default_path = os.path.join(os.path.dirname(__file__), 'data/vocab.pkl')

# Reserved symbols
SYM_PAD = '<pad>'   # padding
SYM_SOS = '<start>' # Start of sentence
SYM_EOS = '<end>'   # End of sentence
SYM_UNK = '<unk>'   # Unknown word

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        # Init mappings between words and ids
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        # Special symbols
        self.pad = SYM_PAD
        self.sos = SYM_SOS
        self.eos = SYM_EOS
        self.unk = SYM_UNK

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[SYM_UNK]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def load_vocab(vocab_path=default_path):
    """Load Vocabulary object from a pickle file"""

    logger.info("Loading vocab from %s"%(vocab_path))
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    logger.info("Vocab size: %d"%(len(vocab)))

    return vocab

def build_vocab(qa_json, threshold):
    """Build a simple vocabulary wrapper."""
    with open(qa_json) as f:
        vg_qas = json.load(f)

    num_questions = 0
    counter = Counter()
    for entry in vg_qas:
        for qa in entry["qas"]:
            question = str(qa["question"])
            tokens = nltk.tokenize.word_tokenize(question.lower())
            counter.update(tokens)

            answer = str(qa["answer"])
            tokens = nltk.tokenize.word_tokenize(answer.lower())
            counter.update(tokens)

            num_questions += 1

            if num_questions % 1000 == 0:
                logger.info("Tokenized %d questions." %(num_questions))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word(SYM_PAD)
    vocab.add_word(SYM_SOS)
    vocab.add_word(SYM_EOS)
    vocab.add_word(SYM_UNK)

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    logging.basicConfig(level=logging.INFO)
    vocab = build_vocab(qa_json=args.caption_path,
                        threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    logger.info("Total vocabulary size: %d" %len(vocab))
    logger.info("Saved the vocabulary wrapper to '%s'" %vocab_path)


if __name__ == '__main__':
    ROOT_DIR = '/home/donlee/QBot/qbot/cnnlstm/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default=ROOT_DIR+'data/question_answers.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default=default_path, 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
