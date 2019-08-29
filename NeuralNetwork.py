import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
from torch.utils.data import DataLoader
from DialogueDataset import DialogueDataset
from Metrics import Metrics

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.patience = 0
        self.init_clip_max_norm = 5.0
        self.optimizer = None
        self.best_result = [0, 0, 0, 0, 0, 0]
        self.metrics = Metrics(self.args.score_file_path)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self):
        raise NotImplementedError


    def train_step(self, i, data):
        with torch.no_grad():
            batch_u, batch_r, batch_y = (item.cuda(device=self.device) for item in data)

        self.optimizer.zero_grad()
        logits = self.forward(batch_u, batch_r)
        loss = self.loss_func(logits, target=batch_y)
        loss.backward()
        self.optimizer.step()
        print('Batch[{}] - loss: {:.6f}  batch_size:{}'.format(i, loss.item(), batch_y.size(0)) )  # , accuracy, corrects
        return loss


    def fit(self, X_train_utterances,  X_train_responses, y_train,
                  X_dev_utterances, X_dev_responses, y_dev):

        if torch.cuda.is_available(): self.cuda()

        dataset = DialogueDataset(X_train_utterances, X_train_responses, y_train)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

        self.loss_func = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.l2_reg)

        for epoch in range(self.args.epochs):
            print("\nEpoch ", epoch+1, "/", self.args.epochs)
            avg_loss = 0

            self.train()
            for i, data in enumerate(dataloader):
                loss = self.train_step(i, data)

                if i > 0 and i % 500 == 0:
                    self.evaluate(X_dev_utterances, X_dev_responses, y_dev)
                    self.train()

                if epoch >= 2 and self.patience >= 3:
                    print("Reload the best model...")
                    self.load_state_dict(torch.load(self.args.save_path))
                    self.adjust_learning_rate()
                    self.patience = 0

                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)

                avg_loss += loss.item()
            cnt = len(y_train) // self.args.batch_size + 1
            print("Average loss:{:.6f} ".format(avg_loss/cnt))
            self.evaluate(X_dev_utterances, X_dev_responses, y_dev)


    def adjust_learning_rate(self, decay_rate=.5):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            self.args.learning_rate = param_group['lr']
        print("Decay learning rate to: ", self.args.learning_rate)


    def evaluate(self, X_dev_utterances, X_dev_responses, y_dev, is_test=False):
        y_pred = self.predict(X_dev_utterances, X_dev_responses)
        with open(self.args.score_file_path, 'w') as output:
            for score, label in zip(y_pred, y_dev):
                output.write(
                    str(score) + '\t' +
                    str(label) + '\n'
                )

        result = self.metrics.evaluate_all_metrics()
        print("Evaluation Result: \n",
              "MAP:", result[0], "\t",
              "MRR:", result[1], "\t",
              "P@1:", result[2], "\t",
              "R1:",  result[3], "\t",
              "R2:",  result[4], "\t",
              "R5:",  result[5])

        if not is_test and result[3] + result[4] + result[5] > self.best_result[3] + self.best_result[4] + self.best_result[5]:
            print("Best Result: \n",
                  "MAP:", self.best_result[0], "\t",
                  "MRR:", self.best_result[1], "\t",
                  "P@1:", self.best_result[2], "\t",
                  "R1:",  self.best_result[3], "\t",
                  "R2:",  self.best_result[4], "\t",
                  "R5:",  self.best_result[5])
            self.patience = 0
            self.best_result = result
            torch.save(self.state_dict(), self.args.save_path)
            print("save model!!!\n")
        else:
            self.patience += 1


    def predict(self, X_dev_utterances, X_dev_responses):
        self.eval()
        y_pred = []
        dataset = DialogueDataset(X_dev_utterances, X_dev_responses)
        dataloader = DataLoader(dataset, batch_size=100)

        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_u, batch_r = (item.cuda() for item in data)

            logits = self.forward(batch_u, batch_r)
            y_pred += logits.data.cpu().numpy().tolist()
        return y_pred


    def load_model(self, path):
        self.load_state_dict(state_dict=torch.load(path))
        if torch.cuda.is_available(): self.cuda()

