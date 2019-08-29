import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from NeuralNetwork import NeuralNetwork


class TransformerBlock(nn.Module):

    def __init__(self, input_size, is_layer_norm=False):
        super(TransformerBlock, self).__init__()
        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        return self.linear2(self.relu(self.linear1(X)))

    def forward(self, Q, K, V, episilon=1e-8):
        '''
        :param Q: (batch_size, max_r_words, embedding_dim)
        :param K: (batch_size, max_u_words, embedding_dim)
        :param V: (batch_size, max_u_words, embedding_dim)
        :return: output: (batch_size, max_r_words, embedding_dim)  same size as Q
        '''
        dk = torch.Tensor([max(1.0, Q.size(-1))]).cuda()

        Q_K = Q.bmm(K.permute(0, 2, 1)) / (torch.sqrt(dk) + episilon)
        Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size, max_r_words, max_u_words)
        V_att = Q_K_score.bmm(V)

        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)  # (batch_size, max_r_words, embedding_dim)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X

        return output


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=1)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, X, mask=None):
        '''
        :param X:
        :param mask:   http://juditacs.github.io/2018/12/27/masked-attention.html
        :return:
        '''
        M = F.tanh(self.linear1(X))  # (batch_size, max_u_words, embedding_dim)
        M = self.linear2(M)
        M[~mask] = float('-inf')
        score = F.softmax(M, dim=1)    # (batch_size, max_u_words, 1)

        output = (score * X).sum(dim=1)  # (batch_size, embedding_dim)
        return output



class MSN(NeuralNetwork):
    '''
        A pytorch version of Sequential Matching Network which is proposed in
            "Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-based Chatbots"
    '''
    def __init__(self, word_embeddings, args):
        self.args = args
        super(MSN, self).__init__()

        self.word_embedding = nn.Embedding(num_embeddings=len(word_embeddings), embedding_dim=200, padding_idx=0,
                                           _weight=torch.FloatTensor(word_embeddings))

        self.alpha = 0.5
        self.gamma = 0.3
        self.selector_transformer = TransformerBlock(input_size=200)
        self.W_word = nn.Parameter(data=torch.Tensor(200, 200, 10))
        self.v = nn.Parameter(data=torch.Tensor(10, 1))
        self.linear_word = nn.Linear(2*50, 1)
        self.linear_score = nn.Linear(in_features=3, out_features=1)

        self.transformer_utt = TransformerBlock(input_size=200)
        self.transformer_res = TransformerBlock(input_size=200)
        self.transformer_ur = TransformerBlock(input_size=200)
        self.transformer_ru = TransformerBlock(input_size=200)

        self.A1 = nn.Parameter(data=torch.Tensor(200, 200))
        self.A2 = nn.Parameter(data=torch.Tensor(200, 200))
        self.A3 = nn.Parameter(data=torch.Tensor(200, 200))

        self.cnn_2d_1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3,3))
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.cnn_2d_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3))
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.cnn_2d_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
        self.maxpooling3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.affine2 = nn.Linear(in_features=3*3*64, out_features=300)

        self.gru_acc = nn.GRU(input_size=300, hidden_size=args.gru_hidden, batch_first=True)
        # self.attention = Attention(input_size=300, hidden_size=300)
        self.affine_out = nn.Linear(in_features=args.gru_hidden, out_features=1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.init_weights()
        print(self)


    def init_weights(self):
        init.uniform_(self.W_word)
        init.uniform_(self.v)
        init.uniform_(self.linear_word.weight)
        init.uniform_(self.linear_score.weight)

        init.xavier_normal_(self.A1)
        init.xavier_normal_(self.A2)
        init.xavier_normal_(self.A3)
        init.xavier_normal_(self.cnn_2d_1.weight)
        init.xavier_normal_(self.cnn_2d_2.weight)
        init.xavier_normal_(self.cnn_2d_3.weight)
        init.xavier_normal_(self.affine2.weight)
        init.xavier_normal_(self.affine_out.weight)
        for weights in [self.gru_acc.weight_hh_l0, self.gru_acc.weight_ih_l0]:
            init.orthogonal_(weights)


    def word_selector(self, key, context):
        '''
        :param key:  (bsz, max_u_words, d)
        :param context:  (bsz, max_u_words, d)
        :return: score:
        '''
        dk = torch.sqrt(torch.Tensor([200])).cuda()
        A = torch.tanh(torch.einsum("blrd,ddh,bud->blruh", context, self.W_word, key)/dk)
        A = torch.einsum("blruh,hp->blrup", A, self.v).squeeze()   # b x l x u x u

        a = torch.cat([A.max(dim=2)[0], A.max(dim=3)[0]], dim=-1) # b x l x 2u
        s1 = torch.softmax(self.linear_word(a).squeeze(), dim=-1)  # b x l
        return s1

    def utterance_selector(self, key, context):
        '''
        :param key:  (bsz, max_u_words, d)
        :param context:  (bsz, max_u_words, d)
        :return: score:
        '''
        key = key.mean(dim=1)
        context = context.mean(dim=2)
        s2 = torch.einsum("bud,bd->bu", context, key)/(1e-6 + torch.norm(context, dim=-1)*torch.norm(key, dim=-1, keepdim=True) )
        return s2

    def distance(self, A, B, C, epsilon=1e-6):
        M1 = torch.einsum("bud,dd,brd->bur", [A, B, C])

        A_norm = A.norm(dim=-1)
        C_norm = C.norm(dim=-1)
        M2 = torch.einsum("bud,brd->bur", [A, C]) / (torch.einsum("bu,br->bur", A_norm, C_norm) + epsilon)
        return M1, M2

    def context_selector(self, context, hop=[1, 2, 3]):
        '''
        :param context: (batch_size, max_utterances, max_u_words, embedding_dim)
        :param key: (batch_size, max_u_words, embedding_dim)
        :return:
        '''
        su1, su2, su3, su4 = context.size()
        context_ = context.view(-1, su3, su4)   # (batch_size*max_utterances, max_u_words, embedding_dim)
        context_ = self.selector_transformer(context_, context_, context_)
        context_ = context_.view(su1, su2, su3, su4)

        multi_match_score = []
        for hop_i in hop:
            key = context[:, 10-hop_i:, :, :].mean(dim=1)
            key = self.selector_transformer(key, key, key)

            s1 = self.word_selector(key, context_)
            s2 = self.utterance_selector(key, context_)
            s = self.alpha * s1 + (1 - self.alpha) * s2
            multi_match_score.append(s)

        multi_match_score = torch.stack(multi_match_score, dim=-1)
        match_score = self.linear_score(multi_match_score).squeeze()
        mask = (match_score.sigmoid() >= self.gamma).float()
        match_score = match_score * mask
        context = context * match_score.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return context

    def get_Matching_Map(self, bU_embedding, bR_embedding):
        '''
        :param bU_embedding: (batch_size*max_utterances, max_u_words, embedding_dim)
        :param bR_embedding: (batch_size*max_utterances, max_r_words, embedding_dim)
        :return: E: (bsz*max_utterances, max_u_words, max_r_words)
        '''
        # M1 = torch.einsum("bud,dd,brd->bur", bU_embedding, self.A1, bR_embedding)  # (bsz*max_utterances, max_u_words, max_r_words)
        M1, M2 = self.distance(bU_embedding, self.A1, bR_embedding)

        Hu = self.transformer_utt(bU_embedding, bU_embedding, bU_embedding)
        Hr = self.transformer_res(bR_embedding, bR_embedding, bR_embedding)
        # M2 = torch.einsum("bud,dd,brd->bur", [Hu, self.A2, Hr])
        M3, M4 = self.distance(Hu, self.A2, Hr)

        Hur = self.transformer_ur(bU_embedding, bR_embedding, bR_embedding)
        Hru = self.transformer_ru(bR_embedding, bU_embedding, bU_embedding)
        # M3 = torch.einsum("bud,dd,brd->bur", [Hur, self.A3, Hru])
        M5, M6 = self.distance(Hur, self.A3, Hru)

        M = torch.stack([M1, M2, M3, M4, M5, M6], dim=1)  # (bsz*max_utterances, channel, max_u_words, max_r_words)
        return M


    def UR_Matching(self, bU_embedding, bR_embedding):
        '''
        :param bU_embedding: (batch_size*max_utterances, max_u_words, embedding_dim)
        :param bR_embedding: (batch_size*max_utterances, max_r_words, embedding_dim)
        :return: (bsz*max_utterances, (max_u_words - width)/stride + 1, (max_r_words -height)/stride + 1, channel)
        '''
        M = self.get_Matching_Map(bU_embedding, bR_embedding)

        Z = self.relu(self.cnn_2d_1(M))
        Z = self.maxpooling1(Z)

        Z = self.relu(self.cnn_2d_2(Z))
        Z =self.maxpooling2(Z)

        Z = self.relu(self.cnn_2d_3(Z))
        Z =self.maxpooling3(Z)

        Z = Z.view(Z.size(0), -1)  # (bsz*max_utterances, *)

        V = self.tanh(self.affine2(Z))   # (bsz*max_utterances, 50)
        return V

    def forward(self, bU, bR):
        '''
        :param bU: batch utterance, size: (batch_size, max_utterances, max_u_words)
        :param bR: batch responses, size: (batch_size, max_r_words)
        :return: scores, size: (batch_size, )
        '''
        # u_mask = (bU != 0).unsqueeze(dim=-1).float()
        # u_mask_sent = ((bU != 0).sum(dim=-1) !=0 ).unsqueeze(dim=-1)
        # r_mask = (bR != 0).unsqueeze(dim=-1).float()

        bU_embedding = self.word_embedding(bU) # + self.position_embedding(bU_pos) # * u_mask
        bR_embedding = self.word_embedding(bR) # + self.position_embedding(bR_pos) # * r_mask
        multi_context = self.context_selector(bU_embedding, hop=[1, 2, 3])

        su1, su2, su3, su4 = multi_context.size()
        multi_context = multi_context.view(-1, su3, su4)   # (batch_size*max_utterances, max_u_words, embedding_dim)

        sr1, sr2, sr3= bR_embedding.size()   # (batch_size, max_r_words, embedding_dim)
        bR_embedding = bR_embedding.unsqueeze(dim=1).repeat(1, su2, 1, 1)  # (batch_size, max_utterances, max_r_words, embedding_dim)
        bR_embedding = bR_embedding.view(-1, sr2, sr3)   # (batch_size*max_utterances, max_r_words, embedding_dim)

        V = self.UR_Matching(multi_context, bR_embedding)
        V = V.view(su1, su2, -1)  # (bsz, max_utterances, 300)

        H, _ = self.gru_acc(V)  # (bsz, max_utterances, rnn2_hidden)
        # L = self.attention(V, u_mask_sent)
        L = self.dropout(H[:,-1,:])

        output = torch.sigmoid(self.affine_out(L))
        return output.squeeze()


