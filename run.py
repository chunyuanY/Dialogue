import time
import argparse
import pickle
from MSN import MSN
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

task = 'ubuntu'
task_dic = {
    'ubuntu':'./dataset/ubuntu_data/',
    'douban':'./dataset/DoubanConversaionCorpus/',
    'alime':'./dataset/E_commerce/'
}

path = task_dic[task]
print("dataset: ", path)
vocab, word_embeddings = pickle.load(file=open(path+"vocab_and_embeddings.pkl", 'rb'))
model_cls = MSN

data_bsz = {
	"ubuntu": 200,
	"douban": 150,
	"alime":  200
}
batch_size = data_bsz[task]


## Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--vocab_size",
                    default=len(vocab),
                    type=int,
                    help="The vocabulary size.")
parser.add_argument("--max_utterances",
                    default=10,
                    type=int,
                    help="The maximum number of utterances.")
parser.add_argument("--max_words",
                    default=50,
                    type=int,
                    help="The maximum number of words for each utterance.")
parser.add_argument("--batch_size",
                    default=batch_size,
                    type=int,
                    help="The batch size.")
parser.add_argument("--rnn1_hidden",
                    default=300,
                    type=int,
                    help="The hidden size of GRU in layer 1")
parser.add_argument("--rnn2_hidden",
                    default=300,
                    type=int,
                    help="The hidden size of GRU in layer 2")
parser.add_argument("--learning_rate",
                    default=1e-3,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--l2_reg",
                    default=0.0,
                    type=float,
                    help="The l2 regularization.")
parser.add_argument("--epochs",
                    default=5,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--save_path",
                    default="./checkpoint/" + task+ '.' + model_cls.__name__ + ".pt",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_path",
                    default=path+"score_file.txt",
                    type=str,
                    help="The path to save model.")
args = parser.parse_args()
print(args)


def train(model_cls):
    X_train_utterances, X_train_responses, y_train = pickle.load(file=open(path+"train.pkl", 'rb'))
    X_dev_utterances, X_dev_responses, y_dev = pickle.load(file=open(path+"test.pkl", 'rb'))

    model = model_cls(word_embeddings, args=args)
    model.fit(
        X_train_utterances,  X_train_responses, y_train,
        X_dev_utterances, X_dev_responses, y_dev
    )

def test(model_cls):
    X_test_utterances, X_test_responses, y_test = pickle.load(file=open(path+"test.pkl", 'rb'))
    model = model_cls(word_embeddings, args=args)
    model.load_model(args.save_path)
    model.evaluate(X_test_utterances, X_test_responses, y_test, is_test=True)

def test_adversarial(model_cls):
    model = model_cls(word_embeddings, args=args)
    model.load_model(args.save_path)
    print("adversarial test set (k=1): ")
    X_test_utterances, X_test_responses, y_test = pickle.load(file=open(path+"test_adversarial_k_1.pkl", 'rb'))
    model.evaluate(X_test_utterances, X_test_responses, y_test, is_test=True)
    print("adversarial test set (k=2): ")
    X_test_utterances, X_test_responses, y_test = pickle.load(file=open(path+"test_adversarial_k_2.pkl", 'rb'))
    model.evaluate(X_test_utterances, X_test_responses, y_test, is_test=True)
    print("adversarial test set (k=3): ")
    X_test_utterances, X_test_responses, y_test = pickle.load(file=open(path+"test_adversarial_k_3.pkl", 'rb'))
    model.evaluate(X_test_utterances, X_test_responses, y_test, is_test=True)

if __name__ == '__main__':
    start = time.time()
    # train(model_cls)
    test(model_cls)
    test_adversarial(model_cls)
    end = time.time()
    print("use time: ", (end-start)/60, " min")





