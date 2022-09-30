import argparse
import cv2
from model import ResidualBlock, Encoder, Generator
from datasets import PairedDataset
from utils import *

# Arguments
parser = argparse.ArgumentParser(description='Test UNIT')

parser.add_argument('--gpu_num', type=int, default=0)

# Training parameters
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=1)

# Model
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--n_sample', type=int, default=2)
parser.add_argument('--n_feats', type=int, default=64)
parser.add_argument('--n_res_blocks', type=int, default=3)

# Dataset
parser.add_argument('--domain1', type=str, default='Dog')
parser.add_argument('--domain2', type=str, default='RealandFake_faces')
parser.add_argument('--train_size', type=int, default=1000)

# Transformations
parser.add_argument('--resize', type=bool, default=True)
parser.add_argument('--crop', type=bool, default=False)
parser.add_argument('--patch_size', type=int, default=229)
parser.add_argument('--flip', type=bool, default=False)
parser.add_argument('--normalize', type=bool, default=True)

args = parser.parse_args()


def Test_UNIT(args):
    # Device
    device = torch.device('cuda:{}'.format(args.gpu_num))

    # Models
    shared_E1 = ResidualBlock()
    shared_E2 = ResidualBlock()
    E1 = Encoder(args.in_channels, args.n_feats, args.n_sample, args.n_res_blocks, shared_block=shared_E1).to(device)
    E2 = Encoder(args.in_channels, args.n_feats, args.n_sample, args.n_res_blocks, shared_block=shared_E2).to(device)

    shared_G1 = ResidualBlock()
    shared_G2 = ResidualBlock()
    G1 = Generator(args.in_channels, args.n_feats, args.n_sample, args.n_res_blocks, shared_block=shared_G1).to(device)
    G2 = Generator(args.in_channels, args.n_feats, args.n_sample, args.n_res_blocks, shared_block=shared_G2).to(device)

    dataset_name = '{}2{}_train_size_{}'.format(args.domain1, args.domain2, args.train_size)
    E1.load_state_dict(torch.load('./checkpoints/{}/E1_{}epochs.pth'.format(dataset_name, args.n_epochs), map_location=device))
    E2.load_state_dict(torch.load('./checkpoints/{}/E2_{}epochs.pth'.format(dataset_name, args.n_epochs), map_location=device))
    G1.load_state_dict(torch.load('./checkpoints/{}/G1_{}epochs.pth'.format(dataset_name, args.n_epochs), map_location=device))
    G2.load_state_dict(torch.load('./checkpoints/{}/G2_{}epochs.pth'.format(dataset_name, args.n_epochs), map_location=device))

    E1.eval()
    E2.eval()
    G1.eval()
    G2.eval()

    # Dataset
    transform = transforms.Compose(get_transforms(args))
    test_dataset = PairedDataset(domain1=args.domain1, domain2=args.domain2, train_size=args.train_size, train=False, transform=transform)

    # Evaluate
    save_dir = './results/{}/{}epochs'.format(dataset_name, args.n_epochs)
    fake_domain1_dir = os.path.join(save_dir, '{}'.format(args.domain1))
    fake_domain2_dir = os.path.join(save_dir, '{}'.format(args.domain2))
    if not os.path.exists(fake_domain1_dir):
        os.makedirs(fake_domain1_dir)
    if not os.path.exists(fake_domain2_dir):
        os.makedirs(fake_domain2_dir)

    for index, data in enumerate(test_dataset):
        if index >= 30:
            break

        real_A, real_B = data['domain1'], data['domain2']
        real_A, real_B = real_A.to(device), real_B.to(device)
        real_A, real_B = torch.unsqueeze(real_A, dim=0), torch.unsqueeze(real_B, dim=0)

        latent_A, latent_B = E1(real_A)['mu'], E2(real_B)['mu']

        fake_B = 0.5*(G2(latent_A)+1.0)
        fake_A = 0.5*(G1(latent_B)+1.0)
        real_A = 0.5*(real_A+1.0)
        real_B = 0.5*(real_B+1.0)

        AtoB = torch.cat([real_A, fake_B], dim=3)
        BtoA = torch.cat([real_B, fake_A], dim=3)

        AtoB, BtoA = torch.squeeze(AtoB).cpu(), torch.squeeze(BtoA).cpu()
        AtoB, BtoA = tensor_to_numpy(AtoB), tensor_to_numpy(BtoA)
        AtoB, BtoA = cv2.cvtColor(AtoB, cv2.COLOR_RGB2BGR), cv2.cvtColor(BtoA, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(fake_domain1_dir, '{}.png'.format(index+1)), BtoA)
        cv2.imwrite(os.path.join(fake_domain2_dir, '{}.png'.format(index+1)), AtoB)


if __name__ == "__main__":
    Test_UNIT(args=args)
