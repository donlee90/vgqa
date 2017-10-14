from ModelWrapper import VQGWrapper, VQAWrapper, QAWrapper
from VisualGenomeQA import Vocabulary
import logging

import os

def generate_topk_questions(img_paths,
        model_config='config.json', batch_size=64):

    """ Generate topk candidate questions using vanilla VQGModel(cnnlstm).
    """

    vqg = VQGWrapper(model_config)

    i = 0            # start of batch
    N = len(img_paths)    # Total number of images
    topk_questions = []

    while i < N:
        batch = img_paths[i:i+batch_size]
        outputs, probs = vqg.predict(batch)
        assert len(outputs) == len(batch)
        topk_questions.extend(outputs)
        i += batch_size

    return topk_questions



def generate_cnnlstm_questions(img_paths,
        model_config='config.json', batch_size=64):
    """ Generate a question for each image using cnnlstm model """

    topk_questions = generate_topk_questions(img_paths,
                                             model_config=model_config,
                                             batch_size=batch_size)
    questions = [topk[0] for topk in topk_questions]

    return questions


def generate_vqg_qa_questions(img_paths,
        model_config='config.json', batch_size=64):

    # Generate topk questions using CNN-LSTM
    topk_questions = generate_topk_questions(img_paths,
                                             model_config=model_config,
                                             batch_size=batch_size)

    # Rank questions with QA Model
    qa = QAWrapper(model_config)
    questions = [qa.rank(topk)[0] for topk in topk_questions]

    return questions


def generate_vqg_vqa_questions(img_paths, p_threshold=0.1,
        model_config='config.json', batch_size=64):

    # Generate topk questions using CNN-LSTM
    topk_questions = generate_topk_questions(img_paths,
                                             model_config=model_config,
                                             batch_size=batch_size)

    assert len(topk_questions) == len(img_paths)

    # Rank questions with QA Model
    qa = QAWrapper(model_config)
    ranked_questions = [qa.rank(topk) for topk in topk_questions]

    vqa = VQAWrapper(model_config)
    uncertain_questions = []
    for img_path, questions in zip(img_paths, ranked_questions):
        preds = vqa.predict(zip([img_path]*len(questions), questions))

        for question, answers, probs in zip(*preds):
            if probs[0] < p_threshold:
                uncertain_questions.append(question)
                break

    return uncertain_questions


if __name__=='__main__':

    logging.basicConfig(level=logging.DEBUG)

    img_dir = 'imgs'

    # Build inputs
    image_paths = []
    for path in sorted(os.listdir(img_dir)):
        image_paths.append(os.path.join(img_dir, path))

    print image_paths

    """
    questions = generate_cnnlstm_questions(image_paths)
    print "\n"
    for q in questions:
        print q

    questions = generate_vqg_qa_questions(image_paths)
    print "\n"
    for q in questions:
        print q
    """

    questions = generate_vqg_vqa_questions(image_paths)
    print "\n"
    for q in questions:
        print q
