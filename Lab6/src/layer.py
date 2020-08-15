import torch
import data
from math import sqrt


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, head_num, hidden_size, output_dim):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.query_matrix = []
        self.key_matrix = []
        self.value_matrix = []

        for i in range(head_num):
            self.query_matrix.append(data.rand_matrix(input_dim, hidden_size))
            self.key_matrix.append(data.rand_matrix(input_dim, hidden_size))
            self.value_matrix.append(data.rand_matrix(input_dim, hidden_size))
        self.weight_matrix = data.rand_matrix(hidden_size * head_num,
                                              hidden_size)

        self.query_matrix = data.list2torch(self.query_matrix,
                                            torch.FloatTensor)
        self.key_matrix = data.list2torch(self.key_matrix, torch.FloatTensor)
        self.value_matrix = data.list2torch(self.value_matrix,
                                            torch.FloatTensor)
        self.weight_matrix = data.list2torch(self.weight_matrix,
                                             torch.FloatTensor)

        self.linear = torch.nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)

        query = torch.matmul(x, self.query_matrix)
        key = torch.matmul(x, self.key_matrix)
        value = torch.matmul(x, self.value_matrix)

        key = key.permute(0, 1, 3, 2)
        score = torch.matmul(query, key) / sqrt(self.hidden_size)
        score = torch.nn.functional.softmax(score, dim=2)
        prediction = torch.matmul(score, value)

        prediction = prediction.permute(0, 2, 3, 1)
        size = list(prediction.size())
        prediction = prediction.reshape(size[0], size[1], size[2] * size[3], 1)
        prediction = prediction.squeeze()

        prediction = torch.matmul(prediction, self.weight_matrix)
        prediction = torch.nn.functional.softmax(prediction, dim=2)

        out = self.linear(prediction)

        return out


class Decoder(torch.nn.Module):
    def __init__(self, input_dim, head_num, hidden_size, output_dim):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.query_matrix = []
        self.key_matrix = []
        self.value_matrix = []

        for i in range(head_num):
            self.query_matrix.append(data.rand_matrix(input_dim, hidden_size))
            self.key_matrix.append(data.rand_matrix(input_dim, hidden_size))
            self.value_matrix.append(data.rand_matrix(input_dim, hidden_size))
        self.weight_matrix = data.rand_matrix(input_dim * head_num,
                                              hidden_size)

        self.query_matrix = data.list2torch(self.query_matrix,
                                            torch.FloatTensor)
        self.key_matrix = data.list2torch(self.key_matrix, torch.FloatTensor)
        self.value_matrix = data.list2torch(self.value_matrix,
                                            torch.FloatTensor)
        self.weight_matrix = data.list2torch(self.weight_matrix,
                                             torch.FloatTensor)

        self.linear = torch.nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)

        query = torch.matmul(x, self.query_matrix)
        key = torch.matmul(x, self.key_matrix)
        value = torch.matmul(x, self.value_matrix)

        key = key.permute(0, 1, 3, 2)
        score = torch.matmul(query, key) / sqrt(self.hidden_size)

        score = torch.tril(input, diagonal=0)
        score = torch.nn.functional.softmax(score, dim=3)
        prediction = torch.matmul(score, value)

        prediction = prediction.permute(0, 2, 3, 1)
        size = list(prediction.size())
        prediction = prediction.reshape(size[0], size[1], size[2] * size[3], 1)
        prediction = prediction.squeeze()

        prediction = torch.matmul(prediction, self.weight_matrix)
        prediction = torch.nn.functional.softmax(prediction, dim=2)

        out = self.linear(prediction)

        return out


class OutputLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(OutputLayer, self).__init__()

        self.linear1 = torch.nn.Linear(input_dim, 1)
        self.linear2 = torch.nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.squeeze(x, 2)
        x = self.linear2(x)
        return torch.nn.functional.softmax(x, dim=1)
