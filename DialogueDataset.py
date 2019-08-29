import torch
from torch.utils.data import TensorDataset



class DialogueDataset(TensorDataset):

    def __init__(self, X_utterances, X_responses, y_labels=None):
        super(DialogueDataset, self).__init__()
        X_utterances = torch.LongTensor(X_utterances)

        X_responses = torch.LongTensor(X_responses)
        print("X_utterances: ", X_utterances.size())
        print("X_responses: ", X_responses.size())

        if y_labels is not None:
            y_labels = torch.FloatTensor(y_labels)
            print("y_labels: ", y_labels.size())
            self.tensors = [X_utterances, X_responses, y_labels]
        else:
            self.tensors = [X_utterances, X_responses]

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return len(self.tensors[0])

