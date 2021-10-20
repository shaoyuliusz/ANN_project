"""
@author: Yuanchu Dang
"""

import numpy as np
import torch
import torch.optim as optim

from torch.autograd import Variable
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize


def get_pretrained_embeddings(model_name, sentences):
    """
    :param model_name: string representing the pretrained model
    :param sentences: a list of strings (sentences)
    :return: a list of sentence embeddings (equal length)
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    return embeddings


def train_word_embeddings(sentence, embedding_size,
                          train_iter = 5,
                          train_body_len = 3,
                          train_ratio = 0.001,
                          train_batch_len = 20):
    """
    :param sentence: a single sentence (string)
    :param embedding_size: how many digits to represent a word
    :param train_iter: how many iterations to run the torch optimzation
    :param train_body_len: train param
    :param train_ratio: train learning rate
    :param train_batch_len: batch size
    :return:
    """

    # Create vocabulary, word lists, and word to index mapping
    word_list = word_tokenize(sentence)
    vocabulary = np.unique(word_list)
    word_to_index = {word: index for index, word in enumerate(vocabulary)}
    word_list_size = len(word_list)
    vocabulary_size = len(vocabulary)

    # Construct co-occurence matrix
    comation = np.zeros((len(vocabulary), len(vocabulary)))
    for i in range(word_list_size):
        for j in range(1, train_body_len + 1):
            index = word_to_index[word_list[i]]
            if i - j > 0:
                left_index = word_to_index[word_list[i - j]]
                comation[index, left_index] += 1.0 / j
            if i + j < word_list_size:
                right_index = word_to_index[word_list[i + j]]
                comation[index, right_index] += 1.0 / j
    co_occurrences = np.transpose(np.nonzero(comation))

    # GET Weight function
    def f_x(Y):
        Xmax = 2
        alpha_value = 0.75
        if Y < Xmax:
            return (Y / Xmax) ** alpha_value
        return 1

    # Set up word vectors and biases, and optimizer
    left_embedded, right_embedded = [
        [Variable(torch.from_numpy(np.random.normal(0, 0.01, (embedding_size, 1))),
                  requires_grad = True) for j in range(vocabulary_size)] for i in range(2)]
    left_biase, right_biase = [
        [Variable(torch.from_numpy(np.random.normal(0, 0.01, 1)),
                  requires_grad = True) for j in range(vocabulary_size)] for i in range(2)]
    improvement = optim.Adam(left_biase + right_biase + left_embedded + right_embedded, lr = train_ratio)

    # Batch sampling function
    def show_batches():
        left_vectors = []
        right_vectors = []
        covals = []
        left_vector_bias = []
        right_vector_bias = []
        samples = np.random.choice(np.arange(len(co_occurrences)), size = train_batch_len, replace = False)
        for example in samples:
            index = tuple(co_occurrences[example])
            left_vectors.append(left_embedded[index[0]])
            right_vectors.append(right_embedded[index[1]])
            covals.append(comation[index])
            left_vector_bias.append(left_biase[index[0]])
            right_vector_bias.append(right_biase[index[1]])
        return left_vectors, right_vectors, covals, left_vector_bias, right_vector_bias

    # Train model
    for term in range(train_iter):
        cnt_batch = int(word_list_size / train_batch_len)
        agv_lossless = 1
        for batch in range(cnt_batch):
            improvement.zero_grad()
            left_vectors, right_vectors, covals, left_vector_bias, right_vector_bias = show_batches()
            loss = sum([torch.mul((torch.dot(left_vectors[i].view(-1), right_vectors[i].view(-1)) +
                                   left_vector_bias[i] + right_vector_bias[i] - np.log(covals[i])) ** 2,
                                  f_x(covals[i])) for i in range(train_batch_len)])
            agv_lossless += loss.data[0] / cnt_batch
            loss.backward()
            improvement.step()
        print("lossless of term on average " + str(term + 1) + ": ", agv_lossless)

    # Assemble embeddings for all words
    all_embeddings = []
    for word_index in range(len(vocabulary)):
        # Create embedding by summing left and right embeddings
        this_embedding = ((left_embedded[word_index].data + right_embedded[word_index].data).numpy())[:, 0]
        all_embeddings.append(this_embedding)
    all_embeddings = np.array(all_embeddings)

    return vocabulary, all_embeddings