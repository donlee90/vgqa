from ModelWrapper import VQGWrapper, VQAWrapper, QAWrapper
from VisualGenomeQA import Vocabulary
import logging

import os
import time

logging.basicConfig(level=logging.DEBUG)

# Build inputs
image_paths = []
for path in os.listdir('temp'):
    image_paths.append(os.path.join('temp', path))

image_paths = image_paths[:20]

# Generate Topk questions for each image
start = time.time()
vqg = VQGWrapper('config.json')
topk_questions = vqg.predict(image_paths)

# For each question predict answer
qa = QAWrapper('config.json')
for questions in topk_questions:
    print questions
    topk_answers = qa.predict(questions)

# Test VQG
vqa = VQAWrapper('config.json')
for i, questions in enumerate(topk_questions):
    image_path = image_paths[i]
    vqa.predict(zip([image_path]*vqg.beam_size, questions))
end = time.time()

print end - start, 'seconds'
