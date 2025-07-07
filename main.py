import argparse
from train import MGSAL
def get_argument_parser():

    parser = argparse.ArgumentParser(description='Ours')
    parser.add_argument('--dataset_root_path', default=r"\dataset", type=str, help='')
    parser.add_argument('--dataset', default="flickr25k", type=str, help='')
    parser.add_argument('--epoch', type=int, default=50,  help='')
    parser.add_argument('--batch_size', default=32, type=int, help='Size of a training mini-batch.')
    parser.add_argument("--k_bits", type=int, default=64, help="length of hash codes.")
    parser.add_argument("--device", type=int, default=0, help="device")
    parser.add_argument("--num_workers", type=int, default=12, help="num_workers")
    parser.add_argument("--save_dir", type=str, default="path", help="Saved model folder")
    parser.add_argument("--TRAIN", type=bool, default="True", help="is TRAIN")
    parser.add_argument("--iter", type=int, default=1, help="Print loss")
    parser.add_argument("--warmup", type=int, default=4, help="")
    parser.add_argument("--feat_lens", type=int, default=512, help="feature lens")
    parser.add_argument("--scal", type=int, default=4, help="scaling parameter")
    parser.add_argument("--IregionK", type=int, default=45 , help="image Region")
    parser.add_argument("--nearK", type=int, default=11, help="low sample k")
    parser.add_argument("--b", type=int, default=0.4, help="f")
    parser.add_argument("--s_intra", type=int, default=0.6, help="similarity matrix S")
    parser.add_argument("--s_inter", type=int, default=0.3, help="similarity matrix S")
    parser.add_argument("--sl", type=int, default=0.7, help="similarity matrix S_l")
    parser.add_argument("--clip_lr", type=float, default=0.000002, help="clip lr")
    parser.add_argument("--lr", type=float, default=0.001, help="lr")
    parser.add_argument("--query_num", type=int, default=2000)
    parser.add_argument("--train_num", type=int, default=5000)
    return parser


def main():
    parser = get_argument_parser()
    opt = parser.parse_args()
    print("warmup:",opt.warmup)
    print("Usedevice:", opt.device)
    print("UseBit:", opt.k_bits)
    print("BS:", opt.batch_size)
    print("IregionK:", opt.IregionK)
    print("nearK:", opt.nearK)
    print("s_intra:", opt.s_intra)
    print("s_inter:", opt.s_inter)
    print("sl:", opt.sl)

    epoch = opt.epoch
    Model = MGSAL(opt)

    if opt.TRAIN == True:
        for epoch in range(epoch):
            Model.train(epoch)
    else:
        Model.eval(epoch, test=True)


if __name__ == '__main__':
    main()
