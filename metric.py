import torch
import torch.nn.functional as F
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self,batch_size, device='cuda:0', temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.register_buffer("temperature", torch.tensor(temperature).to(device))

    def forward(self, emb_i, emb_j,indices=None,W=None):
        self.batch_size = emb_i.shape[0]
        self.negatives_mask = (~torch.eye(self.batch_size * 2, self.batch_size * 2, dtype=bool).to(self.device)).float()
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)
        # (bs, dim)  --->  (bs, dim)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        if W != None:
            W = W.to(self.device).float()
            W = torch.cat((W, W), dim=0)
            loss_partial = loss_partial * W
            num=W.sum(dim=-1)
            loss = torch.sum(loss_partial) / (2 *num)
        else:
            loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss



def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities
def simK(img_emb,cap_emb,nearK):

    bs=img_emb.size(0)
    # basic matching loss
    s_img = img_emb.mm(img_emb.t())
    s_cap = cap_emb.mm(cap_emb.t())


    k =nearK
    if k>bs:
      k=bs
    img_k_indices = []


    for row in s_img:
        _, indices = row.topk(k, largest=True)
        img_k_indices.append(indices.tolist())

    img_k_indices = torch.tensor(img_k_indices, dtype=torch.long)

    cap_k_indices = []

    for row in s_cap:
        _, indices = row.topk(k, largest=True)
        cap_k_indices.append(indices.tolist())
    cap_k_indices = torch.tensor(cap_k_indices, dtype=torch.long)


    result = torch.zeros(img_k_indices.shape[0], dtype=torch.int)


    for i in range(img_k_indices.shape[0]):
        intersection = set(img_k_indices[i].tolist()) & set(cap_k_indices[i].tolist())
        num_same_elements = len(intersection)

        result[i] = num_same_elements
    temp=result

    for i in range(len( temp)):
        if  temp[i] == 1:
            temp[i] = 0
        elif  temp[i] > 1:
            temp[i] = 1


    # W = F.normalize(result.float()-1.0, dim=-1)
    # eW=torch.exp(W)-0.12
    # num = eW.mean(dim=-1)
    indices_k = torch.where(result <= 1)[0]
    return indices_k,temp