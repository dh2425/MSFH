import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution,GCNet_IMG,PositionalEncoding
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from data.model.clip_model.model import  Transformer


class HashingModel(nn.Module):
    """
    Hashing model
    """
    def __init__(self,opt, clip_info=None):
        super().__init__()
        self.batch_size=opt.batch_size
        self.opt=opt
        self.device = self.opt.device
        self.FuseTrans_S=self.FuseTrans = FuseTransEncoder(num_layers=2, hidden_size=1024, nhead=4).to(self.device)
        self.feat_lens=512
        self.nbits=opt.k_bits
        self.ImageMlp = ImageMlp(self.feat_lens, self.nbits).to(self.device)
        self.TextMlp = TextMlp(self.feat_lens, self.nbits).to(self.device)

    def forward(self, img_tokens, txt_tokens, img_cls, txt_eos, key_padding_mask):
        output_dict = {}
        img_tokens=img_tokens.transpose(1,0)
        txt_tokens=txt_tokens.transpose(1,0)

        img_region, txt_region = self.region(img_tokens, txt_tokens, key_padding_mask)
        img_region, txt_region,all_region=self.FuseTrans_S(img_region, txt_region)

        img_tokens_mean = img_tokens.mean(dim=1)
        txt_tokens_mean = txt_tokens.mean(dim=1)
        img_weight = torch.matmul(img_tokens, txt_tokens_mean.unsqueeze(1).permute(0, 2, 1)).squeeze(2)
        txt_weight = torch.matmul(txt_tokens, img_tokens_mean.unsqueeze(1).permute(0, 2, 1)).squeeze(2)
        img_weight = img_weight / img_weight.sum(dim=1, keepdim=True)
        txt_weight = txt_weight / txt_weight.sum(dim=1, keepdim=True)
        img_tokens_mean = img_tokens * img_weight.unsqueeze(-1)
        txt_tokens_mean = txt_tokens * txt_weight.unsqueeze(-1)
        img_tokens_mean = img_tokens_mean.sum(dim=1)
        txt_tokens_mean = txt_tokens_mean.sum(dim=1)

        img_tokens_fu,txt_tokens_fu,_=self.FuseTrans(img_tokens_mean,txt_tokens_mean)

        img_embedding = img_tokens_fu*(1-self.opt.b) + img_cls*self.opt.b
        text_embedding = txt_tokens_fu*(1-self.opt.b) + txt_eos*self.opt.b

        output_dict['img_tokens_fu'] = img_tokens_fu
        output_dict['txt_tokens_fu'] = txt_tokens_fu

        output_dict['img_cls'] = img_cls
        output_dict['txt_eos'] = txt_eos

        output_dict['img_region_fus'] = img_region
        output_dict['txt_region_fus'] = txt_region
        output_dict['all_region_fus'] = all_region

        img_embedding = F.normalize(img_embedding, dim=-1)
        text_embedding = F.normalize(text_embedding, dim=-1)


        output_dict['img_embedding'] = img_embedding
        output_dict['text_embedding'] = text_embedding

        d_img_token_Mlp = self.ImageMlp(img_embedding)
        d_txt_token_Mlp = self.TextMlp(text_embedding)

        output_dict['d_img_idk_Mlp'] = d_img_token_Mlp
        output_dict['d_txt_idk_Mlp'] = d_txt_token_Mlp

        return output_dict



    def region(self, img, cap, cap_mask, img_len=None):
        i2t_sim_mean = []
        t2i_sim_mean = []
        #bs
        s_seq_batch = cap.size(0)
        s_seq_len = cap.size(1)

        # if k>cap.shape[1]:
        #     k=cap.shape[1]
        s_len_mask = cap_mask.unsqueeze(2).expand(s_seq_batch, s_seq_len, cap.shape[2])
        cap.masked_fill_(s_len_mask, value=0)
        # cap = cap[:, 1:, :]
        alignments = torch.matmul(img, cap.permute(0, 2, 1))
        # aggr_row_img = alignments.max(2)[0]

        aggr_row_img = alignments.mean(dim=-1)
        aggr_clm_cap = alignments.permute(0, 2, 1).mean(dim=-1)
        # aggr_row_img = alignments.max(dim=2)[0]
        # aggr_clm_cap = alignments.permute(0, 2, 1).max(dim=2)[0]

        i2t_sim_mean.append(aggr_row_img.mean(dim=-1))
        t2i_sim_mean.append(aggr_clm_cap.mean(dim=-1))

        dim = 1
        results = []
        for idx in range(aggr_clm_cap.size(0)):
            length = sum(cap_mask[idx] == False)
            # length = length - 1
            tmp = torch.split(aggr_clm_cap[idx], split_size_or_sections=length, dim=dim - 1)[
                0]  # engths[idx] 当前图像region数量 #（region[idx]_length,1024�?
            # if k>length:
            #     temp_k=int(length/2)
            # else:
            #     temp_k=k
            # temp_k = 32
            # C_tmp, C_index_tmp  = aggr_clm_cap.topk(k=temp_k, dim=-1)
            temp_k = int(length * 0.9)
            C_tmp, C_index_tmp = tmp.topk(k=temp_k, dim=-1)
            cap_i = cap[idx, C_index_tmp]
            avg_i = cap_i.mean(dim - 1)
            results.append(avg_i)

        # construct with the batch
        cap_k = torch.stack(results, dim=0)
        # C, C_index = aggr_clm_cap.topk(k=k, dim=-1)

         #k = int(49 / 2)
        k=30
        I, I_index = aggr_row_img.topk(k=k, dim=-1)

        img_k = img[torch.arange(I_index.shape[0]).unsqueeze(1), I_index]
        img_k = img_k.mean(dim=1)
        # cap_k = cap[torch.arange(C_index.shape[0]).unsqueeze(1), C_index]

        cap_k=cap.mean(dim=1)
        return img_k, cap_k



class FuseTransEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nhead):  # num_layers, self.token_size, nhead = 2, 1024, 4
        super(FuseTransEncoder, self).__init__()
        # encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=False)
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformerEncoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.d_model = hidden_size
        self.sigal_d = int(self.d_model / 2)

    def forward(self, img, txt):  # torch.Size([1, 128, 1024])
        temp_tokens = torch.cat((img, txt), dim=1)  # torch.Size([128, 1024])
        tokens = temp_tokens.unsqueeze(0)  # torch.Size([1, 128, 1024])
        encoder_X = self.transformerEncoder(tokens)  # torch.Size([1, 128, 1024])
        encoder_X_r = encoder_X.reshape(-1, self.d_model)  # torch.Size([128, 1024])
        # encoder_X_r = F.normalize(encoder_X_r, p=2, dim=-1)
        encoder_X_r = F.normalize(encoder_X_r,dim=-1)
        img, txt = encoder_X_r[:, :self.sigal_d], encoder_X_r[:, self.sigal_d:]
        return img, txt,encoder_X_r


class  FuseTransEncoder_dim0(nn.Module):
    def __init__(self, num_layers, hidden_size, nhead):  # num_layers, self.token_size, nhead = 2, 1024, 4
        super(FuseTransEncoder_dim0, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformerEncoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.d_model = hidden_size
        self.sigal_d = int(self.d_model / 2)

    def forward(self, img, txt):  # torch.Size([1, 128, 1024])
        temp_tokens = torch.cat((img, txt), dim=0)  # torch.Size([128, 1024])
        tokens = temp_tokens.unsqueeze(0)  # torch.Size([1, 128, 1024])
        encoder_X = self.transformerEncoder(tokens)  # torch.Size([1, 128, 1024])
        encoder_X_r = encoder_X.reshape(-1, self.d_model)  # torch.Size([128, 1024])
        # encoder_X_r = F.normalize(encoder_X_r, p=2, dim=-1)
        encoder_X_r = F.normalize(encoder_X_r,dim=-1)
        img, txt = encoder_X_r[:, :self.sigal_d], encoder_X_r[:, self.sigal_d:]
        return img, txt,encoder_X_r
class ImageMlp(nn.Module):
    def __init__(self, input_dim, hash_lens, dim_feedforward=[1024, 128, 1024], dropout=0.1):  # input_dim=512
        super(ImageMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(4096, hash_lens)
        self.tanh = nn.Tanh()
    def _ff_block(self, x):
        x = F.normalize(x, p=2, dim=1)
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        out = self.tanh(hid)
        return out

    def forward(self, X):  # torch.Size([128, 512])
        mlp_output = self._ff_block(X)
        mlp_output = F.normalize(mlp_output, p=2, dim=1)
        return mlp_output

class TextMlp(nn.Module):
    def __init__(self, input_dim, hash_lens, dim_feedforward=[1024, 128, 1024], dropout=0.1):
        super(TextMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(4096, hash_lens)
        self.tanh = nn.Tanh()

    def _ff_block(self, x):
        x = F.normalize(x, p=2, dim=1)
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        out = self.tanh(hid)
        return out

    def forward(self, X):
        mlp_output = self._ff_block(X)
        mlp_output = F.normalize(mlp_output, p=2, dim=1)
        return mlp_output





def similarity(img_cls, txt_eos,img_region, txt_region,all_region,opt):
        img_cls = F.normalize(img_cls, dim=1)
        txt_eos = F.normalize(txt_eos, dim=1)
        img_region= F.normalize(img_region, dim=1)
        txt_region= F.normalize(txt_region, dim=1)

        sigma = 1.
        n_img = img_cls.size(0)
        n_cap = txt_eos.size(0)
        with torch.no_grad():
            batch_sim_t2t = torch.matmul(txt_eos, txt_eos.t())
            batch_sim_i2i = torch.matmul(img_cls, img_cls.t())
            batch_sim_i2t = torch.matmul(batch_sim_t2t,batch_sim_i2i.t())
            batch_sim_t2i = torch.matmul(batch_sim_i2i,batch_sim_t2t.t())

            batch_sim_all = torch.matmul(all_region, all_region.t())

            region_i2i= torch.matmul(img_region, img_region.t())
            region_t2t = torch.matmul(txt_region, txt_region.t())

            # # Boolean type matrix returned by logical judgment
            batch_t2t_connect = (batch_sim_t2t - batch_sim_t2t.topk(k=int(n_cap * opt.s_intra), dim=1, largest=True)[
                                                     0][:, -1:]) >= 0
            batch_i2i_connect = (batch_sim_i2i - batch_sim_i2i.topk(k=int(n_img * opt.s_intra), dim=1, largest=True)[
                                                     0][:, -1:]) >= 0
            k = int(n_img * opt.s_inter)
            if k <= 0:
                k = 1
            batch_i2t_connect = (batch_sim_i2t - batch_sim_i2t.topk(k=k, dim=1, largest=True)[0][:, -1:]) >= 0
            batch_t2i_connect = (batch_sim_t2i - batch_sim_t2i.topk(k=k, dim=1, largest=True)[0][:, -1:]) >= 0
            batch_all_connect = (batch_sim_all - batch_sim_all.topk(k=k, dim=1, largest=True)[0][:, -1:]) >= 0

        mask = batch_t2t_connect * batch_i2i_connect
        batch_i2i_relation = torch.exp(-torch.cdist(img_region, img_region) / sigma) *batch_i2i_connect
        batch_t2t_relation = torch.exp(-torch.cdist(txt_region, txt_region) / sigma) * batch_t2t_connect
        batch_i2t_relation = torch.exp(-torch.cdist(region_i2i, region_t2t) / sigma) * batch_i2t_connect
        batch_t2i_relation = torch.exp(-torch.cdist(region_t2t, region_i2i) / sigma) * batch_t2i_connect
        batch_all_relation = torch.exp(-torch.cdist(all_region,all_region) / sigma) * batch_all_connect

        S = batch_i2i_relation * 0.4 + batch_t2t_relation * 0.4 + (batch_i2t_relation*0.5 + batch_t2i_relation*0.5 ) * 0.1+batch_all_relation* 0.1
        return  S

def similarity_l(all_fu,opt):
        all_fu= F.normalize(all_fu, dim=1)
        sigma = 1.
        n_all_fu = all_fu.size(0)
        with torch.no_grad():
            batch_sim = torch.matmul(all_fu, all_fu.t())
            k = int(n_all_fu* opt.sl)
            if k <= 0:
                k = 1
            batch_connect = (batch_sim  - batch_sim .topk(k=k, dim=1, largest=True)[0][:, -1:]) >= 0
        batch_relation = torch.exp(-torch.cdist(all_fu, all_fu) / sigma) *  batch_connect
        S = batch_relation
        return S













