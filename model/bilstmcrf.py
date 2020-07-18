# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-01-05 23:15:17

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.bilstm import BiLSTM
from model.crf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, data):
        super(BiLSTM_CRF, self).__init__()
        print( "build batched lstmcrf...")
        self.gpu = data.HP_gpu
        ## add two more label for downlayer lstm, use original label size for CRF
        label_size = data.label_alphabet_size
        data.label_alphabet_size += 2
        self.crf = CRF(label_size, self.gpu)

        label_size_ner = data.label_alphabet_size_ner
        data.label_alphabet_size_ner += 2
        self.crf_ner = CRF(label_size_ner, self.gpu)

        label_size_general = data.label_alphabet_size_general
        data.label_alphabet_size_general += 2
        self.crf_general = CRF(label_size_general, self.gpu)

        self.lstm = BiLSTM(data)


    def neg_log_likelihood_loss(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths,  char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask):
        outs = self.lstm.get_output_score(gaz_list, word_inputs, biword_inputs, word_seq_lengths,  char_inputs, char_seq_lengths, char_seq_recover)
        total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        return total_loss, tag_seq 


    def neg_log_likelihood_loss_ner(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths,  char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask):
        outs = self.lstm.get_output_score_ner(gaz_list, word_inputs, biword_inputs, word_seq_lengths,  char_inputs, char_seq_lengths, char_seq_recover)
        total_loss = self.crf_ner.neg_log_likelihood_loss(outs, mask, batch_label)
        scores, tag_seq = self.crf_ner._viterbi_decode(outs, mask)
        return total_loss, tag_seq 

    def neg_log_likelihood_loss_general(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths,  char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask):
        outs = self.lstm.get_output_score_general(gaz_list, word_inputs, biword_inputs, word_seq_lengths,  char_inputs, char_seq_lengths, char_seq_recover)
        total_loss = self.crf_general.neg_log_likelihood_loss(outs, mask, batch_label)
        scores, tag_seq = self.crf_general._viterbi_decode(outs, mask)
        return total_loss, tag_seq 


    def forward(self, is_ner, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
        if not is_ner:
            outs = self.lstm.get_output_score(gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            outs = self.lstm.get_output_score_ner(gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
            scores, tag_seq = self.crf_ner._viterbi_decode(outs, mask)
        return tag_seq


    # def get_lstm_features(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
    #     return self.lstm.get_lstm_features(gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
    #     
