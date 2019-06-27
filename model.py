import torch
import torch.nn as nn
from torch.autograd import Variable



class LanguageModel(nn.Module):
    def __init__(self, config):
        super(LanguageModel, self).__init__()
        self.config = config
        self.hidden_dim = self.config['num_hidden_units']
        self.num_layers = self.config['num_layers']

        self.lstm = nn.LSTM(input_size=self.config['embedding_size'],
                                                hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=self.config['dropout'])

        self.fc = nn.Linear(self.hidden_dim, config['question_feature'])


    def forward(self, x):
                
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim))
            c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim))
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:,-1,:])

        return out


class VQA_FeatureModel(nn.Module):
    def __init__(self, config):
        super(VQA_FeatureModel, self).__init__()
        self.conig = config

        self.question_model = LanguageModel(config)

        self.bn1 = nn.BatchNorm1d(config['image_feature'])
        self.bn2 = nn.BatchNorm1d(config['question_feature']) 
        
        self.dropout = nn.Dropout(config['dropout']) 
        self.fc_image = nn.Linear(512, config['image_feature'])    # ResNet-18 feature vector has dimensions: 512

        self.fc_combined = nn.Linear( 256, config['vocab_size']) #config['image_feature']+config['question_feature'] , config['vocab_size'])


    def forward(self, x1, x2):
        
        
        image_feature = self.fc_image(x1)
        question_feature = self.question_model(x2)

        image_feature = torch.nn.functional.relu(image_feature)
        question_feature = torch.nn.functional.relu(question_feature)

        image_feature = self.bn1(image_feature)
        question_feature = self.bn2(question_feature) 

        concat = torch.mul(image_feature, question_feature)
        #concat = torch.cat((image_feature, question_feature), dim=1)
        concat = self.dropout(concat)        

        out = self.fc_combined(concat)

        return out

