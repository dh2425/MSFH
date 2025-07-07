import time
import os
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.model.clip_model.model import load_download_clip
from data.load_data import generate_dataset
from torch.utils.data import DataLoader
from model import HashingModel,similarity,similarity_l
from metric import ContrastiveLoss,simK
from optimization import BertAdam
from evluation import calc_map_k,get_code


class Modal(nn.Module):
    def __init__(self,opt):
        super(Modal, self).__init__()
        self.clip_path='./cache/ViT-B-32.pt' # help="pretrained clip path."
        self.clip, clip_info = load_download_clip(self.clip_path)
        self.hash = HashingModel(opt,clip_info=clip_info)


    def forward(self, image, text, key_padding_mask):
        img_tokens, _, img_cls = self.clip.encode_image(image)
        txt_tokens, _, new_key_padding_mask, txt_eos = self.clip.encode_text(text, key_padding_mask)
        output_dict = self.hash(img_tokens, txt_tokens, img_cls, txt_eos, new_key_padding_mask)
        return output_dict

class MGSAL:
    def __init__(self, opt,):
        self.opt=opt
        dataset = opt.dataset
        index_file = "index.mat"
        caption_file = "caption.mat"
        label_file = "label.mat"
        index_file = os.path.join(opt.dataset_root_path, dataset, index_file)  # './dataset\\flickr25k\\index.mat'
        caption_file = os.path.join(opt.dataset_root_path, dataset, caption_file)  # './dataset\\flickr25k\\caption.mat'
        label_file = os.path.join(opt.dataset_root_path, dataset, label_file)  # './dataset\\flickr25k\\label.mat'
        print(caption_file)

        train_data, query_data, retrieval_data = generate_dataset(captionFile=caption_file,
                                                                  indexFile=index_file,
                                                                  labelFile=label_file,
                                                                  maxWords=32,
                                                                  imageResolution=224,
                                                                  query_num=opt.query_num,
                                                                  train_num=opt.train_num,
                                                                  seed=1)

        self.train_labels = train_data.get_all_label().float()  # (10000,24)
        self.query_labels = query_data.get_all_label().float()  # (5000,24)
        self.retrieval_labels = retrieval_data.get_all_label().float()  # (15015,24)
        retrieval_num = len(self.retrieval_labels)
        self.opt.retrieval_num = retrieval_num

        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            pin_memory=False,
            shuffle=True
        )

        self.query_loader = DataLoader(
            dataset=query_data,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            pin_memory=False,
            shuffle=True
        )
        self.retrieval_loader = DataLoader(
            dataset=retrieval_data,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            pin_memory=False,
            shuffle=True
        )


        self.model = Modal(opt).to(opt.device)
        self.model.float()

        self.optimizer = BertAdam(
            [
                {'params': self.model.clip.parameters(), 'lr': self.opt.clip_lr },
                {'params': self.model.hash.parameters(), 'lr': self.opt.lr},
            ],
            lr=self.opt.lr, warmup=0.05, schedule='warmup_cosine',
            b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.opt.epoch,
            weight_decay=0.01, max_grad_norm=1.0
        )

        self.loss_l2 = torch.nn.MSELoss()
        self.closs= ContrastiveLoss(self.opt.batch_size)
        self.max_map = {'i2t': 0, "t2i": 0}
        self.best_epoch = 0

    def load_checkpoints(self):
        self.model.load_state_dict(torch.load("path/model.pth", map_location=f"cuda:{self.opt.device}"))
        return self.model

    def save_model(self,datase,epoch):
        # torch.save(self.model.state_dict(), os.path.join(self.opt.save_dir, f"model_{datase}_{epoch}.pth"))
        # timestamp = int(time.time())
        # _{timestamp}
        torch.save(self.model.state_dict(), os.path.join(self.opt.save_dir, f"model_{self.opt.dataset}_{self.opt.k_bits}.pth"))

    def train(self,epoch):

            iter = 0
            self.model.train()
            print("####################### Train epochs: %d #######################" % epoch)
            for image, text, key_padding_mask, _, index in self.train_loader:
                image = image.float().to(self.opt.device, non_blocking=True)
                text = text.to(self.opt.device, non_blocking=True)
                key_padding_mask = key_padding_mask.to(self.opt.device, non_blocking=True)
                output_dict = self.model(image, text, key_padding_mask)
                loss=self.modal_loss(output_dict,epoch)
                iter += 1
                if iter%100==0:
                    print(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch+1)%self.opt.iter==0:
                self.eval(epoch)


    def modal_loss(self,output_dict,epoch):

        cod_img= output_dict['d_img_idk_Mlp']
        cod_txt = output_dict['d_txt_idk_Mlp']

        img_embedding = output_dict['img_embedding']
        text_embedding = output_dict['text_embedding']

        img_cls = output_dict['img_cls']
        txt_eos = output_dict['txt_eos']

        img_region_fus = output_dict['img_region_fus']
        txt_region_fus = output_dict['txt_region_fus']
        all_region_fus = output_dict['all_region_fus']

        all_fu = torch.cat((img_embedding, text_embedding), 1)
        S_embedding = similarity_l(all_fu, self.opt) *self.opt.scal

        S = similarity(img_cls, txt_eos, img_region_fus, txt_region_fus, all_region_fus, self.opt) * self.opt.scal

        loss1 = self.closs(img_embedding, text_embedding)

        loss3 =loss4=loss5= 0
        B = torch.sign(cod_img.detach() + cod_txt.detach())
        loss6 = F.mse_loss(cod_img, B) / cod_img.shape[0] / self.opt.k_bits + F.mse_loss(cod_txt, B) / \
                cod_img.shape[0] / self.opt.k_bits
        if (epoch + 1) >= self.opt.warmup:
            indices, W = simK(img_embedding, text_embedding, self.opt.nearK)
            loss2 = self.closs(cod_img, cod_txt, indices, W)
        else:
            loss2 = self.closs(cod_img, cod_txt)

        if (epoch + 1) >= self.opt.warmup:
            BI_BI = cod_img.mm(cod_img.t())
            BT_BT = cod_txt.mm(cod_txt.t())
            BI_BT = cod_img.mm(cod_txt.t())
            BT_BI = cod_txt.mm(cod_txt.t())
            loss3 = F.mse_loss(BI_BI, S) + F.mse_loss(BT_BT, S)
            loss4 = F.mse_loss(BI_BT, S) + F.mse_loss(BT_BI, S) - (cod_img * cod_txt).sum(dim=1).mean()
            loss5 = self.loss_l2(S, S_embedding)
        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        return loss


    def eval(self,epoch,test=False):
        print("TEST")
        self.model.eval()
        k_bits = self.opt.k_bits

        retrieval_num = self.opt.retrieval_num
        q_i, q_t = get_code(self.model, self.query_loader, k_bits, self.opt.device, self.opt.query_num)
        r_i, r_t = get_code(self.model, self.retrieval_loader, k_bits,self.opt.device, retrieval_num)
        _k_ = None
        mAPi2t = calc_map_k(q_i.to(self.opt.device), r_t.to(self.opt.device), self.query_labels.to(self.opt.device), self.retrieval_labels.to(self.opt.device),
                            _k_).item()
        mAPt2i = calc_map_k(q_t.to(self.opt.device), r_i.to(self.opt.device), self.query_labels.to(self.opt.device), self.retrieval_labels.to(self.opt.device),
                            _k_).item()
        print("mAPi2t :", mAPi2t)
        print("mAPt2i :", mAPt2i)

        if mAPi2t + mAPt2i > self.max_map['i2t'] + self.max_map['t2i'] and not test:
            self.best_epoch = epoch
            self.max_map['i2t'] = mAPi2t
            self.max_map['t2i'] = mAPt2i
            self.save_model(self.opt.dataset,epoch)
            self.save_mat(q_i, q_t, r_i, r_t)
        print("best_epoch :",self.best_epoch)
        print("max_mAPi2t :", self.max_map['i2t'])
        print("max_mAPt2i :", self.max_map['t2i'])

    def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt):

        save_dir = os.path.join(self.opt.save_dir, "PR_curve")
        os.makedirs(save_dir, exist_ok=True)

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.cpu().detach().numpy()
        retrieval_labels = self.retrieval_labels.cpu().detach().numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }

        scio.savemat(
            os.path.join(save_dir, f"MSFH-" + self.opt.dataset + "-"+ str(self.opt.k_bits) + ".mat"),
            result_dict)

    def test(self, epoch, test=False):
        from utils.Search_image import search_code, search_calc_map_k

        print("TEST")
        self.model.eval()
        k_bits = self.opt.k_bits
        retrieval_num = self.opt.retrieval_num
        q_i, q_t, q_index = search_code(self.model, self.query_loader, k_bits, self.opt.device, self.opt.query_num)
        r_i, r_t, r_index = search_code(self.model, self.retrieval_loader, k_bits, self.opt.device, retrieval_num)
        _k_ = None
        self.save_mat(q_i, q_t, r_i, r_t)
        mAPi2t = calc_map_k(q_i.to(self.opt.device), r_t.to(self.opt.device), self.query_labels.to(self.opt.device),
                            self.retrieval_labels.to(self.opt.device),
                            _k_).item()
        mAPt2i = calc_map_k(q_t.to(self.opt.device), r_i.to(self.opt.device), self.query_labels.to(self.opt.device),
                            self.retrieval_labels.to(self.opt.device),
                            _k_).item()
        print("mAPi2t :", mAPi2t)
        print("mAPt2i :", mAPt2i)
        mAPi2t = search_calc_map_k(q_i.to(self.opt.device), r_t.to(self.opt.device),
                                   self.query_labels.to(self.opt.device),
                                   self.retrieval_labels.to(self.opt.device), q_index, r_index, _k_).item()
        mAPt2i = search_calc_map_k(q_t.to(self.opt.device), r_i.to(self.opt.device),
                                   self.query_labels.to(self.opt.device),
                                   self.retrieval_labels.to(self.opt.device), q_index, r_index, _k_).item()

        print("mAPi2t :", mAPi2t)
        print("mAPt2i :", mAPt2i)

