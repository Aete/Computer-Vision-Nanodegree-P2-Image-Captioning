import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.dropout = nn.Dropout(0.2)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.batch_n = nn.BatchNorm1d(embed_size, momentum = 0.01)

        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.batch_n(self.dropout(self.embed(features)))
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1,drop_prob = 0.2):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, dropout=0, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        embeddings = self.embed(captions[:,:-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        x, (h,c) = self.lstm(embeddings)
        outputs = self.linear(x)
        return outputs
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        samples = []
        for i in range(max_len):
            hiddens, states = self.lstm(inputs,states)
            output = self.linear(hiddens)
            predicted_value = torch.argmax(output, dim=2)
            predicted_index = predicted_value.item()
            samples.append(predicted_index)
            
            if predicted_index == 1:
                break
            inputs = self.embed(predicted_value)
        return samples
            