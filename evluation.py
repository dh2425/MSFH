
import torch






def calc_neighbor(a: torch.Tensor, b: torch.Tensor): #（10000 24） （bs 24）
    return (a.matmul(b.transpose(0, 1)) > 0).float()


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH

def calc_map_k(qB, rB, query_label, retrieval_label, k=None):
    num_query = query_label.shape[0] #5000
    map = 0.
    if k is None:
        k = retrieval_label.shape[0]
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)#1到4729步长为1
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0  #gand中不为0的索引。
        map += torch.mean(count / tindex)
    map = map / num_query
    return map



def get_code( model,data_loader,k_bits, device,length: int):
    k_bits = k_bits
    rank=0
    img_buffer = torch.empty(length, k_bits, dtype=torch.float).to(device)
    text_buffer = torch.empty(length, k_bits, dtype=torch.float).to(device)

    with torch.no_grad():
        for image, text, key_padding_mask, label, index in data_loader:

            image = image.float().to(rank, non_blocking=True)
            text = text.to(rank, non_blocking=True)
            label= label.to(rank, non_blocking=True)
            key_padding_mask = key_padding_mask.to(rank, non_blocking=True)
            index = index.numpy()
            output_dict = model(image, text, key_padding_mask)


            d_img_idk_Mlp = output_dict['d_img_idk_Mlp']
            d_txt_idk_Mlp = output_dict['d_txt_idk_Mlp']

            img_embedding_Mlp = torch.sign(d_img_idk_Mlp)
            text_embedding_Mlp = torch.sign(d_txt_idk_Mlp)

            # if max(index)>10000:
            #     print(index)

            img_buffer[index, :] = img_embedding_Mlp
            text_buffer[index, :] = text_embedding_Mlp

    return img_buffer, text_buffer