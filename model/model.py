import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers import AutoModel

class BaseModel(nn.Module):
    """_summary_
    베이스라인 모델입니다.
    """
    def __init__(self, model_name, num_labels, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        
        self.model = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.model.config.hidden_size, self.num_labels)
        )

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooler = self.model(input_ids=input_ids, attention_mask=attention_mask).to_tuple()
        logits = self.regressor(pooler)
        
        return logits
    
class T5Model(nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask, tgt_input_ids:None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels = tgt_input_ids)
        return outputs
    
class PLMRNNModel(nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels

        if 't5' in model_name:
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).get_encoder()
        else:
            self.model = AutoModel.from_pretrained(model_name)
        self.lstm = nn.LSTM(input_size=self.model.config.hidden_size,
                            hidden_size=self.model.config.hidden_size,
                            num_layers=3,
                            bidirectional=True,
                            batch_first=True)
        
        self.gru = nn.GRU(input_size=self.model.config.hidden_size,
                          hidden_size=self.model.config.hidden_size,
                          num_layers=3,
                          batch_first=True,
                          bidirectional=True)
        
        self.activation = nn.Tanh()
        self.regressor = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.model.config.hidden_size*2, self.num_labels))
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        _, hidden = self.gru(outputs) 
        # _, (hidden, _) = self.lstm(outputs) 
        outputs = torch.cat([hidden[-1], hidden[-2]], dim=1)
        outputs= self.activation(outputs)
        logits = self.regressor(outputs)

        return logits