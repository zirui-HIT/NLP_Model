import torch
import data
from module import ESIM
import torch.utils.data as Data


def train(train_data_path,
          vocabulary_path,
          embedding_matrix_path,
          save_path,
          data_size=None,
          hidden_dim=50,
          batch_size=1000,
          lr=0.2,
          epoch=30):
    sentence1, sentence2, label = data.load_data(train_data_path, data_size)

    vocabulary = data.load_vocabulary(vocabulary_path)

    embedding_matrix = data.load_embedding(embedding_matrix_path)
    embedding_matrix = data.list2torch(embedding_matrix)

    sentence1, max_length1 = data.tokenizer(sentence1, vocabulary)
    sentence2, max_length2 = data.tokenizer(sentence2, vocabulary)
    max_length = max(max_length1, max_length2)

    sentence1 = data.padding(sentence1, max_length)
    sentence2 = data.padding(sentence2, max_length)
    sentence1 = data.list2torch(sentence1)
    sentence2 = data.list2torch(sentence2)

    label = data.label2num(label)
    label = data.list2torch(label)

    torch_dataset = Data.TensorDataset(sentence1, sentence2, label)
    loader = Data.DataLoader(dataset=torch_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2)

    esim = ESIM(hidden_dim, len(vocabulary), embedding_matrix,
                embedding_matrix.size()[1])
    optimizer = torch.optim.Adam(esim.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    for e in range(epoch):
        for step, (s1, s2, label) in enumerate(loader):
            _, out = esim(s1, s2)
            loss = loss_func(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(esim, save_path)
    return esim
