import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torch.nn.init import normal_
import torch


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ===== MovieLens - NCF by KCH =====


def activation_layer(activation_name='relu'):
    """
    Construct activation layers
    
    Args:
        activation_name: str, name of activation function
        emb_dim: int, used for Dice activation
    Return:
        activation: activation layer
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation_name.lower() == 'tanh':
            activation = nn.Tanh()
        elif activation_name.lower() == 'relu':
            activation = nn.ReLU()
        elif activation_name.lower() == 'leakyrelu':
            activation = nn.LeakyReLU()
        elif activation_name.lower() == 'none':
            activation = None
    elif issubclass(activation_name, BaseModel):
        activation = activation_name()
    else:
        raise NotImplementedError("activation function {} is not implemented".format(activation_name))

    return activation


class MLPLayers(BaseModel):
    """
    여러 층의 MLP Layer Class
    
    Args:
        - layers: (List) input layer, hidden layer, output layer의 node 수를 저장한 List.
                ex) [5, 4, 3, 2] -> input layer: 5 nodes, output layer: 2 nodes, hidden layers: 4 nodes, 3 nodes
        - dropout: (float) dropout 확률
        - activation: (str) activation function의 함수. Default: 'relu'
    Shape:
        - Input: (torch.Tensor) input features. Shape: (batch size, # of input nodes)
        - Output: (torch.Tensor) output features. Shape: (batch size, # of output nodes)
    """
    def __init__(self, layers, dropout, activation='relu'):
        super(MLPLayers, self).__init__()
        
        # initialize Class attributes
        self.layers = layers
        self.n_layers = len(self.layers) - 1
        self.dropout = dropout
        self.activation = activation
        
        # define layers
        mlp_modules = list()
        for i in range(self.n_layers):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            input_size = self.layers[i]
            output_size = self.layers[i+1]
            mlp_modules.append(nn.Linear(input_size, output_size))
            activation_function = activation_layer(self.activation)
            if activation_function is not None:
                mlp_modules.append(activation_function)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        
        self.apply(self._init_weights)
        
    # initialize weights
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    
    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


class KCH_MovieModel_NCF(BaseModel):
    """
    Neural Collaborative Filtering
    
    Args:
        - n_users: (int) 전체 유저의 수
        - n_items: (int) 전체 아이템의 수
        - emb_dim: (int) Embedding의 Dimension
        - layers: (List) Neural CF Layers의 각 node 수를 저장한 List.
                ex) [5, 4, 3, 2] -> hidden layers: 5 nodes, 4 nodes, 3 nodes, 2 nodes
        - dropout: (float) dropout 확률
    Shape:
        - Input: (torch.Tensor) input features, (user_id, item_id). Shape: (batch size, 2)
        - Output: (torch.Tensor) expected implicit feedback. Shape: (batch size,)
    """
    def __init__(self, n_users, n_items, emb_dim, layers, dropout):
        super(KCH_MovieModel_NCF, self).__init__()
        
        # initialize Class attributes
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.layers = layers
        self.n_layers = len(self.layers) + 1
        self.dropout = dropout
        
        # define layers
        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)
        self.mlp_layers = MLPLayers([2 * self.emb_dim] + self.layers, self.dropout)
        self.predict_layer = nn.Linear(self.layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
        
        self.apply(self._init_weights)
        
    # initialize weights
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)
        elif isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    
    def forward(self, input_feature):
        user, item = torch.split(input_feature, [1, 1], -1)
        user = user.squeeze(-1)
        item = item.squeeze(-1)
        
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        
        input_feature = torch.cat((user_e, item_e), -1)
        mlp_output = self.mlp_layers(input_feature)
        output = self.predict_layer(mlp_output)
        output = self.sigmoid(output)
        return output.squeeze(-1)