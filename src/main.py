# import re
# import torch
# from torch import nn
# import random
# import time
# from torch import optim
# import time
# import math
# import torch.nn.functional as F
# from RMC import RMC
# from RMC import RMCCell
# 
# SOS_token = 0
# EOS_token = 1
# 
# MAX_LENGTH = 50000
# device = "cuda"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
# hidden_size = 200
# 
# def asMinutes(s):
#     m = math.floor(s / 60)
#     s -= m * 60
#     return '%dm %ds' % (m, s)
# 
# def timeSince(since, percent):
#     now = time.time()
#     s = now - since
#     es = s / (percent)
#     rs = es - s
#     return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
# 
# class Lang:
#     def __init__(self, name):
#         self.name = name
#         self.word2index = {}
#         self.word2count = {}
#         self.index2word = {0: "SOS", 1: "EOS"}
#         self.n_words = 2  # Count SOS and EOS
# 
#     def addSentence(self, sentence):
#         for word in sentence.split(' '):
#             self.addWord(word)
# 
#     def addWord(self, word):
#         if word not in self.word2index:
#             self.word2index[word] = self.n_words
#             self.word2count[word] = 1
#             self.index2word[self.n_words] = word
#             self.n_words += 1
#         else:
#             self.word2count[word] += 1
#             
# 
# def normalizeString(s):
#     s = s.lower().strip()
#     s = re.sub(r"([.!?])", r" \1", s)
#     s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
#     return s
# 
# def readLangs(lang1, lang2, lines,reverse=False):
#     # Split every line into pairs and normalize
#     pairs = [[s.strip() for s in l.split('\t')] for l in lines]
# 
#     # Reverse pairs, make Lang instances
#     if reverse:
#         pairs = [list(reversed(p)) for p in pairs]
#         input_lang = Lang(lang2)
#         output_lang = Lang(lang1)
#     else:
#         input_lang = Lang(lang1)
#         output_lang = Lang(lang2)
# 
#     return input_lang, output_lang, pairs
# 
# def prepareData(lang1, lang2, filename, reverse=False):
#     lines = [x for x in read_scan(filename)]
#     input_lang, output_lang, pairs = readLangs(lang1, lang2,lines, reverse)
#     print("Read %s sentence pairs" % len(pairs))
#     print("Trimmed to %s sentence pairs" % len(pairs))
#     print("Counting words...")
#     for pair in pairs:
#         input_lang.addSentence(pair[0])
#         output_lang.addSentence(pair[1])
#     print("Counted words:")
#     print(input_lang.name, input_lang.n_words)
#     print(output_lang.name, output_lang.n_words)
#     return input_lang, output_lang, pairs
# 
# class EncoderRNN(nn.Module):
#     def __init__(self, input_size, hidden_size,recurrent_layer,init_hidden=None):
#         super(EncoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.embedding = nn.Embedding(input_size, hidden_size)
#         self.recurrent = recurrent_layer
#         self._init_hidden = init_hidden
# 
#     def forward(self, input, hidden):
#         embedded = self.embedding(input).view(1, 1, -1)
#         output = embedded
#         output, hidden = self.recurrent(output, hidden)
#         return output, hidden
#     
#     def initHidden(self):
#         if self._init_hidden is not None:
#             return self._init_hidden
#         return torch.zeros(1, 1, self.hidden_size, device=device)
# 
# class DecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size,recurrent_layer,init_hidden=None):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.recurrent = recurrent_layer
#         self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax()
#         self._init_hidden = init_hidden
# 
#     def forward(self, input, hidden):
#         output = self.embedding(input).view(1, 1, -1)
#         output = F.relu(output)
#         output, hidden = self.recurrent(output, hidden)
#         output = self.softmax(self.out(output[0]))
#         return output, hidden
# 
#     def initHidden(self):
#         if self._init_hidden is not None:
#             return self._init_hidden
#         return torch.zeros(1, 1, self.hidden_size, device=device)
# 
# def indexesFromSentence(lang, sentence):
#     return [lang.word2index[word] for word in sentence.split(' ')]
# 
# 
# def tensorFromSentence(lang, sentence):
#     indexes = indexesFromSentence(lang, sentence)
#     indexes.append(EOS_token)
#     return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
# 
# 
# def tensorsFromPair(pair,input_lang,output_lang):
#     input_tensor = tensorFromSentence(input_lang, pair[0])
#     target_tensor = tensorFromSentence(output_lang, pair[1])
#     return (input_tensor, target_tensor)
# 
# teacher_forcing_ratio = 0.5
# 
# 
# def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
#     encoder_hidden = encoder.initHidden()
# 
#     encoder_optimizer.zero_grad()
#     decoder_optimizer.zero_grad()
# 
#     input_length = input_tensor.size(0)
#     target_length = target_tensor.size(0)
# 
#     encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
# 
#     loss = 0
# 
#     for ei in range(input_length):
#         encoder_output, encoder_hidden = encoder(
#             input_tensor[ei], encoder_hidden)
#         encoder_outputs[ei] = encoder_output[0, 0]
# 
#     decoder_input = torch.tensor([[SOS_token]], device=device)
# 
#     decoder_hidden = encoder_hidden
# 
#     use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
# 
#     if use_teacher_forcing:
#         # Teacher forcing: Feed the target as the next input
#         for di in range(target_length):
#             decoder_output, decoder_hidden = decoder(
#                 decoder_input, decoder_hidden)
#             #target_indicator = torch.zeros(output_lang.n_words,dtype=torch.long).to(device)
#             #target_indicator[target_tensor[di]] = 1
#             loss += criterion(decoder_output.unsqueeze(0), target_tensor[di])
#             decoder_input = target_tensor[di]  # Teacher forcing
# 
#     else:
#         # Without teacher forcing: use its own predictions as the next input
#         for di in range(target_length):
#             decoder_output, decoder_hidden = decoder(
#                 decoder_input, decoder_hidden)
#             topv, topi = decoder_output.topk(1)
#             decoder_input = topi.squeeze().detach()  # detach from history as input
#             #target_indicator = torch.zeros(output_lang.n_words,dtype=torch.long).to(device)
#             #target_indicator[target_tensor[di]] = 1
#             loss += criterion(decoder_output.unsqueeze(0),target_tensor[di])
#             if decoder_input.item() == EOS_token:
#                 break
# 
#     loss.backward()
# 
#     encoder_optimizer.step()
#     decoder_optimizer.step()
# 
#     return loss.item() / target_length
# 
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')
# import matplotlib.ticker as ticker
# import numpy as np
# 
# 
# def showPlot(points):
#     plt.figure()
#     fig, ax = plt.subplots()
#     # this locator puts ticks at regular intervals
#     loc = ticker.MultipleLocator(base=0.2)
#     ax.yaxis.set_major_locator(loc)
#     plt.plot(points)
# 
# def trainIters(encoder, decoder, pairs,n_iters, input_lang,output_lang,print_every=1000, plot_every=100, learning_rate=0.01):
#     start = time.time()
#     plot_losses = []
#     print_loss_total = 0  # Reset every print_every
#     plot_loss_total = 0  # Reset every plot_every
# 
#     encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
#     decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
#     training_pairs = [tensorsFromPair(random.choice(pairs),input_lang,output_lang)
#                       for i in range(n_iters)]
#     criterion = nn.NLLLoss()
# 
#     for iter in range(1, n_iters + 1):
#         training_pair = training_pairs[iter - 1]
#         input_tensor = training_pair[0]
#         target_tensor = training_pair[1]
# 
#         loss = train(input_tensor, target_tensor, encoder,
#                      decoder, encoder_optimizer, decoder_optimizer, criterion)
#         print_loss_total += loss
#         plot_loss_total += loss
# 
#         if iter % print_every == 0:
#             print_loss_avg = print_loss_total / print_every
#             print_loss_total = 0
#             print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
#                                          iter, iter / n_iters * 100, print_loss_avg))
# 
#         if iter % plot_every == 0:
#             plot_loss_avg = plot_loss_total / plot_every
#             plot_losses.append(plot_loss_avg)
#             plot_loss_total = 0
# 
#     showPlot(plot_losses)
#     
# def evaluate(encoder, decoder, sentence,input_lang,output_lang, max_length=MAX_LENGTH):
#     with torch.no_grad():
#         input_tensor = tensorFromSentence(input_lang, sentence)
#         input_length = input_tensor.size()[0]
#         encoder_hidden = encoder.initHidden()
# 
#         encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
# 
#         for ei in range(input_length):
#             encoder_output, encoder_hidden = encoder(input_tensor[ei],
#                                                      encoder_hidden)
#             encoder_outputs[ei] += encoder_output[0, 0]
# 
#         decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
# 
#         decoder_hidden = encoder_hidden
# 
#         decoded_words = []
#         decoder_attentions = torch.zeros(max_length, max_length)
# 
#         for di in range(max_length):
#             decoder_output, decoder_hidden, decoder_attention = decoder(
#                 decoder_input, decoder_hidden, encoder_outputs)
#             decoder_attentions[di] = decoder_attention.data
#             topv, topi = decoder_output.data.topk(1)
#             if topi.item() == EOS_token:
#                 decoded_words.append('<EOS>')
#                 break
#             else:
#                 decoded_words.append(output_lang.index2word[topi.item()])
# 
#             decoder_input = topi.squeeze().detach()
# 
#         return decoded_words, decoder_attentions[:di + 1]
#     
# def read_scan(filename):
#     with open(filename) as fl:
#         for line in fl:
#             inp = line.split('OUT:')[0].split('IN:')[1]
#             out = line.split('OUT:')[1]
#             yield inp +'\t' + out
# 
# import os
# input_lang,output_lang, pairs = prepareData('eng', 'comm', '../../length_split/tasks_train_length.txt')
# #recurrent_encoder = nn.GRU(hidden_size, hidden_size)
# #recurrent_decoder = nn.GRU(hidden_size,hidden_size)
# rmccell_encoder = RMCCell(hidden_size,1,1,device=device)
# #recurrent_encoder = RMC(rmccell_encoder)
# rmccell_decoder = RMCCell(hidden_size,1,1,device=device)
# #recurrent_decoder = RMC(rmccell_decoder)
# 
# encoder = EncoderRNN(input_lang.n_words,hidden_size,rmccell_encoder,init_hidden=rmccell_encoder.initial_state(1)).to(device)
# decoder = DecoderRNN(hidden_size,output_lang.n_words,rmccell_decoder,init_hidden=rmccell_decoder.initial_state(1)).to(device)
# trainIters(encoder, decoder, pairs,10000, input_lang,output_lang,print_every=50,learning_rate=0.01)
# 
# torch.save(encoder,'enc_rmc.th')
# torch.save(decoder,'dec_rmc.th')