'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# define your epsilon for laplace smoothing here

def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # For each word w,
    # it counts how many times w occurs with each tag in the training data.
    # When processing the test data, it consistently gives w the tag that was seen most often.
    # For unseen words, it should guess the tag that's seen the most often in training dataset.

    # iterate the training dataset
        # update dic = { word: { tag: count } }
    # find the tag with most count in the word -> assign to seen word
    # find the tag with most count -> assign to unseen
    word_tag_count_dic = {}
    tag_count_dic = {}
    
    for train_sentence in train:
        train_sentence.pop(0)
        train_sentence.pop()
        for word_tag_pair in train_sentence:
            # remove START and END
            word = word_tag_pair[0]
            tag = word_tag_pair[1]

            # maintain word tag dic
            if word_tag_count_dic.get(word) is None:
                word_tag_count_dic[word] = {}
                word_tag_count_dic[word][tag] = 1
            else:
                if word_tag_count_dic[word].get(tag) is None:
                    word_tag_count_dic[word][tag] = 1
                else:
                    word_tag_count_dic[word][tag] += 1

            # maintain tag dic
            if tag_count_dic.get(tag) is None:
                tag_count_dic[tag] = 1
            else:
                tag_count_dic[tag] += 1

    word_max_count_dic = { key: max(val, key=val.get) for key, val in word_tag_count_dic.items() }
    tag_max_count = max(tag_count_dic, key=tag_count_dic.get)
    
    test_sentences_tags = []
    for test_sentence in test:
        sentence_tag = []
        for word in test_sentence:
            if word_max_count_dic.get(word) is None:
                sentence_tag.append((word, tag_max_count))
            else:
                sentence_tag.append((word, word_max_count_dic[word]))
        test_sentences_tags.append(sentence_tag)

    return test_sentences_tags

def get_max_viterbi(viterbi, transition_prob, observation_prob, current_time_step, states_prob, current_state, current_word):
    max_viterbi = - math.inf
    max_viterbi_state = None
    for previous_state_index, previous_state in enumerate(states_prob.keys()):
        temp_viterbi = viterbi[previous_state_index][current_time_step - 1] * transition_prob[previous_state][current_state] * observation_prob[current_state][current_word]
        
        if temp_viterbi > max_viterbi:
            max_viterbi = temp_viterbi
            max_viterbi_state = previous_state
    return max_viterbi, max_viterbi_state

def viterbi(train, test):
    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]

    Initial probabilities (How often does each tag occur at the start of a sentence?)
    Transition probabilities (How often does tag 𝑡𝑏 follow tag 𝑡𝑎?)
    Emission(Observation) probabilities (How often does tag t yield word w?)
    '''

    # Count occurrences of tags, tag pairs, tag/word pairs.
    # Compute smoothed probabilities
    # Take the log of each probability
    # Construct the trellis. Notice that for each tag/time pair, you must store not only the probability of the best path but also a pointer to the previous tag/time pair in that path.
    # Return the best path through the trellis.

    # count occurrence of tags
    states_dic = {}
    tag_pair_dic = {}
    tag_word_dic = {}
    for train_sentence in train:

        for word_idx, word_tag_pair in enumerate(train_sentence):
            # remove START and END
            word = word_tag_pair[0]
            tag = word_tag_pair[1]

            # maintain tag dic for state prob
            if states_dic.get(tag) is None:
                states_dic[tag] = 1
            else:
                states_dic[tag] += 1

            # maintain tag pair dic for transition prob
            if tag_pair_dic.get(tag) is None:
                tag_pair_dic[tag] = {}
                # get next tag
                if word_idx + 1 < len(train_sentence):
                    next_tag = train_sentence[word_idx + 1][1]
                    tag_pair_dic[tag][next_tag] = 1
            else:
                if word_idx + 1 < len(train_sentence):
                    next_tag = train_sentence[word_idx + 1][1]
                    if tag_pair_dic[tag].get(next_tag) is None:
                        tag_pair_dic[tag][next_tag] = 1
                    else:
                        tag_pair_dic[tag][next_tag] += 1
            # maintain tag_word_dic for observation prob
            if tag_word_dic.get(tag) is None:
                tag_word_dic[tag] = {}
                tag_word_dic[tag][word] = 1
            else:
                if tag_word_dic[tag].get(word) is None:
                    tag_word_dic[tag][word] = 1
                else:
                    tag_word_dic[tag][word] += 1

    k = 1E-4
    # convert states_dic to state prob
    all_states_count = sum(states_dic.values())
    # print("all_states_count = ", all_states_count)
    # states_prob = {}
    states_prob = defaultdict(lambda: (k))
    for key in states_dic:    
        # states_prob[key] = math.log(states_dic[key] / all_states_count)
        states_prob[key] = (states_dic[key] / all_states_count)

    # Laplace smoothing:
    # 𝑃(Token=OOV|Class=𝑦) = 𝑘 / (# tokens of any word in texts of class 𝑦) + 𝑘 × (# of word types+1)
    # convert tag_pair_dic to transition prob
    transition_prob = {}
    states_list = list(states_prob.keys())
    states_len = len(states_list)
    for current_tag in tag_pair_dic:    
        transition_dic = tag_pair_dic[current_tag]
        all_next_tags_count = sum(transition_dic.values())
        oov_smoothing = k / (states_len + k * (states_len + 1))
        temp_transition_prob = defaultdict(lambda: (oov_smoothing))
        for next_tag in transition_dic:
            # temp_transition_prob[next_tag] = math.log(transition_dic[next_tag] / all_next_tags_count)
            temp_transition_prob[next_tag] = (transition_dic[next_tag] / all_next_tags_count)
        transition_prob[current_tag] = temp_transition_prob

    # Laplace smoothing :
    # 𝑃(Token=𝑥|Class=𝑦)= ((# tokens of word 𝑥 in texts of class 𝑦)+𝑘) / ((# tokens of any word in texts of class 𝑦)+𝑘×(# of word types+1))
    # 𝑃(Token=OOV|Class=𝑦) = 𝑘 / (# tokens of any word in texts of class 𝑦) + 𝑘 × (# of word types+1)
    # convert tag_word_dic to observation prob 
    observation_prob = defaultdict(None)
    for tag in tag_word_dic:
        observation_dic = tag_word_dic[tag]
        all_words_count = sum(observation_dic.values())
        oov_smoothing = k / (states_dic[tag] + k * (len(states_dic.keys()) + 1))
        temp_observation_prob = defaultdict(lambda: (oov_smoothing))
        for word in observation_dic:
            temp_observation_prob[word] = (observation_dic[word] + k) / (states_dic[tag] + k * (len(states_dic.keys()) + 1))
        observation_prob[tag] = temp_observation_prob

    # initialize
    # Q: how many states should I initialize?
    # P(start) = 1 for start state, and 0 for other states
    # previous_state = states_prob['START'] * observation_prob['START']['START']

    # print("transition_prob['START'] = ", transition_prob['START'])
    test_sentences_tags = []
    for test_sentence in test:
        words_len = len(test_sentence)
        viterbi_path_matrix = [[0 for i in range(words_len)] for j in range(states_len)]        
        backpointer_matrix = [[None for i in range(words_len)] for j in range(states_len)]
        # initialize
        first_word = test_sentence[0]
        for state_index, state in enumerate(states_list):
            viterbi_path_matrix[state_index][0] = transition_prob['START'][state] * observation_prob[state][first_word]
            backpointer_matrix[state_index][0] = 'START'
        # break
        for time_step in range(1, words_len):
            for current_state_index, current_state in enumerate(states_list):
                viterbi_value, backtrack_state = get_max_viterbi(viterbi_path_matrix, transition_prob, observation_prob, time_step, states_prob, current_state, test_sentence[time_step])
                viterbi_path_matrix[current_state_index][time_step] = viterbi_value
                backpointer_matrix[current_state_index][time_step] = backtrack_state
        
        # look for best state at time = word_len
        last_state_viterbi = - math.inf
        last_state = None
        for state_index, state in enumerate(states_list):
            temp_viterbi = viterbi_path_matrix[state_index][words_len - 1]
            if temp_viterbi > last_state_viterbi:
                last_state_viterbi = temp_viterbi
                last_state = backpointer_matrix[state_index][words_len - 1]
        
        # look for the best path from backpointer_matrix
        # tags_path = ['END']
        tags_path = []
        tags_path.append(last_state)
        # TODO:
        # 1. there might be some off by one error when backtracing
        # 2. the backtracing logic might be wrong
        for i in range(words_len - 1): # 0 - 31 => words_len = 32
            # print("last_state = ", last_state)
            # print("finding last word at ", words_len - i - 1)
            last_state_index = states_list.index(last_state)
            last_state = backpointer_matrix[last_state_index][words_len - i - 1] # 32 - 0 - 1
            tags_path.append(last_state)

        tags_path.reverse()
        # print("tags_path = ", tags_path)
        word_tag_pairs = list(zip(test_sentence, tags_path))
        print("word_tag_pairs = ", word_tag_pairs)
        test_sentences_tags.append(word_tag_pairs)

        break

    # print("test_sentences_tags = ", test_sentences_tags)
    # raise NotImplementedError("You need to write this part!")
    return test_sentences_tags


def viterbi_ec(train, test):
    '''
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    raise NotImplementedError("You need to write this part!")



