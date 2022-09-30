import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools
import time
import pandas as pd
import matplotlib.pyplot as plt

from model import ResidualBlock, Encoder, Generator, Discriminator
from datasets import PairedDataset
from utils import *


class TrainUNIT:
    def __init__(self, args):
        # Arguments
        self.args = args

        # Device
        self.gpu_num = args.gpu_num
        self.device = torch.device('cuda:{}'.format(self.gpu_num) if torch.cuda.is_available() else 'cpu')

        # Models
        shared_E = ResidualBlock()
        self.E1 = Encoder(args.in_channels, args.n_feats, args.n_sample, args.n_res_blocks, shared_block=shared_E).to(self.device)
        self.E2 = Encoder(args.in_channels, args.n_feats, args.n_sample, args.n_res_blocks, shared_block=shared_E).to(self.device)

        shared_G = ResidualBlock()
        self.G1 = Generator(args.in_channels, args.n_feats, args.n_sample, args.n_res_blocks, shared_block=shared_G).to(self.device)
        self.G2 = Generator(args.in_channels, args.n_feats, args.n_sample, args.n_res_blocks, shared_block=shared_G).to(self.device)

        self.D1 = Discriminator(args.in_channels, args.n_feats).to(self.device)
        self.D2 = Discriminator(args.in_channels, args.n_feats).to(self.device)

        self.E1.apply(weights_init_normal)
        self.E2.apply(weights_init_normal)
        self.G1.apply(weights_init_normal)
        self.G2.apply(weights_init_normal)
        self.D1.apply(weights_init_normal)
        self.D2.apply(weights_init_normal)

        # Training Parameters
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size

        self.lambda0 = args.lambda0
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3
        self.lambda4 = args.lambda4

        # Loss
        self.criterion_MSE = torch.nn.MSELoss()
        self.criterion_L1 = torch.nn.L1Loss()

        # Optimizer
        self.optimizer_G = optim.Adam(
            itertools.chain(self.E1.parameters(), self.E2.parameters(), self.G1.parameters(), self.G2.parameters()),
            lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_D1 = optim.Adam(self.D1.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_D2 = optim.Adam(self.D2.parameters(), lr=args.lr, betas=(0.5, 0.999))

        # Scheduler
        self.scheduler_G = optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=LambdaLR(
            args.n_epochs, args.offset_epochs, args.decay_epochs).step)
        self.scheduler_D1 = optim.lr_scheduler.LambdaLR(self.optimizer_D1, lr_lambda=LambdaLR(
            args.n_epochs, args.offset_epochs, args.decay_epochs).step)
        self.scheduler_D2 = optim.lr_scheduler.LambdaLR(self.optimizer_D2, lr_lambda=LambdaLR(
            args.n_epochs, args.offset_epochs, args.decay_epochs).step)

        # Transform
        transform = transforms.Compose(get_transforms(args))

        # Dataset
        self.dataset = PairedDataset(domain1=args.domain1, domain2=args.domain2, train_size=args.train_size, train=True, transform=transform)
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True)

        # Save Paths
        self.project_name = '{}2{}_train_size_{}'.format(args.domain1, args.domain2, args.train_size)
        self.checkpoint_path = './checkpoints/{}/'.format(self.project_name)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.result_path = './results/{}/'.format(self.project_name)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

    def train(self):
        print(self.device)

        # Losses
        GAN1_losses, GAN2_losses = [], []
        VAE_KL1_losses, VAE_KL2_losses, VAE_Recon1_losses, VAE_Recon2_losses = [], [], [], []
        Cycle_KL1_losses, Cycle_KL2_losses, Cycle_Recon1_losses, Cycle_Recon2_losses = [], [], [], []
        D1_losses, D2_losses = [], []
        G_total_losses, D_total_losses = [], []

        start = time.time()
        for epoch in range(1, self.n_epochs + 1):

            # Training
            for batch, data in enumerate(self.dataloader):
                real_1, real_2 = data['domain1'], data['domain2']
                real_1, real_2 = real_1.to(self.device), real_2.to(self.device)

                self.optimizer_G.zero_grad()
                self.optimizer_D1.zero_grad()
                self.optimizer_D2.zero_grad()

                # Outputs
                mu1, z1 = self.E1(real_1)['mu'], self.E1(real_1)['z']
                mu2, z2 = self.E2(real_2)['mu'], self.E2(real_2)['z']

                recon_1, recon_2 = self.G1(z1), self.G2(z2)
                fake_1, fake_2 = self.G1(z2), self.G2(z1)

                fake_mu1, fake_z1 = self.E1(fake_1)['mu'], self.E1(fake_1)['z']
                fake_mu2, fake_z2 = self.E2(fake_2)['mu'], self.E2(fake_2)['z']

                cycle_1, cycle_2 = self.G1(fake_z2), self.G2(fake_z1)

                pred_1, pred_2 = self.D1(fake_1), self.D2(fake_2)

                # Targets
                target_real = torch.ones_like(pred_1, requires_grad=False).to(self.device)
                target_fake = torch.zeros_like(pred_1, requires_grad=False).to(self.device)

                # Encoder, Generator Losses
                # 1. GAN loss
                loss_GAN_1 = self.criterion_MSE(pred_1, target_real)
                loss_GAN_2 = self.criterion_MSE(pred_2, target_real)

                # 2. VAE loss
                loss_KL_1 = self.compute_kl(mu1)  # regularization
                loss_KL_2 = self.compute_kl(mu2)  # regularization
                loss_ID_1 = self.criterion_L1(recon_1, real_1)  # reconstruction
                loss_ID_2 = self.criterion_L1(recon_2, real_2)  # reconstruction

                # 3. Cycle loss
                loss_KL_1_fake = self.compute_kl(fake_mu1)
                loss_KL_2_fake = self.compute_kl(fake_mu2)
                loss_cycle_1 = self.criterion_L1(cycle_1, real_1)
                loss_cycle_2 = self.criterion_L1(cycle_2, real_2)

                # Total loss
                loss_G = (self.lambda0 * (loss_GAN_1 + loss_GAN_2)
                          + self.lambda1 * (loss_KL_1 + loss_KL_2)
                          + self.lambda2 * (loss_ID_1 + loss_ID_2)
                          + self.lambda3 * (loss_KL_1_fake + loss_KL_2_fake)
                          + self.lambda4 * (loss_cycle_1 + loss_cycle_2))

                loss_G.backward()
                self.optimizer_G.step()

                # Discriminator Losses
                loss_D1 = self.criterion_MSE(self.D1(real_1), target_real) + self.criterion_MSE(self.D1(fake_1.detach()), target_fake)
                loss_D2 = self.criterion_MSE(self.D2(real_2), target_real) + self.criterion_MSE(self.D2(fake_2.detach()), target_fake)

                loss_D1.backward()
                loss_D2.backward()
                self.optimizer_D1.step()
                self.optimizer_D2.step()

                # Save Losses
                GAN1_losses.append(loss_GAN_1.item())
                GAN2_losses.append(loss_GAN_2.item())
                VAE_KL1_losses.append(loss_KL_1.item())
                VAE_KL2_losses.append(loss_KL_2.item())
                VAE_Recon1_losses.append(loss_ID_1.item())
                VAE_Recon2_losses.append(loss_ID_2.item())
                Cycle_KL1_losses.append(loss_KL_1_fake.item())
                Cycle_KL2_losses.append(loss_KL_2_fake.item())
                Cycle_Recon1_losses.append(loss_cycle_1.item())
                Cycle_Recon2_losses.append(loss_cycle_2.item())
                G_total_losses.append(loss_G.item())
                D1_losses.append(loss_D1.item())
                D2_losses.append(loss_D2.item())
                D_total_losses.append(loss_D1.item() + loss_D2.item())

                print(
                    '[Epoch {}][{}/{}] | loss: G_total={:.3f} GAN={:.3f} VAE={:.3f} Cycle={:.3f} D_total={:.3f}'.format(
                        epoch, (batch + 1) * self.batch_size, len(self.dataset), loss_G.item(),
                        (loss_GAN_1.item() + loss_GAN_2.item()),
                        (loss_KL_1.item() + loss_KL_2.item() + loss_ID_1.item() + loss_ID_2.item()),
                        (loss_KL_1_fake.item() + loss_KL_2_fake.item() + loss_cycle_1.item() + loss_cycle_2.item()),
                        (loss_D1.item() + loss_D2.item())
                    ))

            self.scheduler_G.step()
            self.scheduler_D1.step()
            self.scheduler_D2.step()

            # Checkpoints
            if epoch % 100 == 0 or epoch == self.n_epochs:
                torch.save(self.E1.state_dict(), os.path.join(self.checkpoint_path,
                                                              'E1_{}epochs.pth'.format(epoch)))
                torch.save(self.E2.state_dict(), os.path.join(self.checkpoint_path,
                                                              'E2_{}epochs.pth'.format(epoch)))
                torch.save(self.G1.state_dict(), os.path.join(self.checkpoint_path,
                                                              'G1_{}epochs.pth'.format(epoch)))
                torch.save(self.G2.state_dict(), os.path.join(self.checkpoint_path,
                                                              'G2_{}epochs.pth'.format(epoch)))
                torch.save(self.D1.state_dict(), os.path.join(self.checkpoint_path,
                                                              'D1_{}epochs.pth'.format(epoch)))
                torch.save(self.D2.state_dict(), os.path.join(self.checkpoint_path,
                                                              'D2_{}epochs.pth'.format(epoch)))

        # Visualize Loss
        fig, axs = plt.subplots(5, 3)
        axs[0, 0].plot(pd.DataFrame(GAN1_losses))
        axs[0, 0].set_title('GAN1 Loss')
        axs[0, 1].plot(pd.DataFrame(GAN2_losses))
        axs[0, 1].set_title('GAN2 Loss')
        axs[1, 0].plot(pd.DataFrame(VAE_KL1_losses))
        axs[1, 0].set_title('VAE-KL1 Loss')
        axs[1, 1].plot(pd.DataFrame(VAE_KL2_losses))
        axs[1, 1].set_title('VAE-KL2 Loss')
        axs[2, 0].plot(pd.DataFrame(VAE_Recon1_losses))
        axs[2, 0].set_title('VAE-Recon1 Loss')
        axs[2, 1].plot(pd.DataFrame(VAE_Recon2_losses))
        axs[2, 1].set_title('VAE-Recon2 Loss')
        axs[3, 0].plot(pd.DataFrame(Cycle_KL1_losses))
        axs[3, 0].set_title('Cycle-KL1 Loss')
        axs[3, 1].plot(pd.DataFrame(Cycle_KL2_losses))
        axs[3, 1].set_title('Cycle-KL2 Loss')
        axs[4, 0].plot(pd.DataFrame(Cycle_Recon1_losses))
        axs[4, 0].set_title('Cycle-Recon1 Loss')
        axs[4, 1].plot(pd.DataFrame(Cycle_Recon2_losses))
        axs[4, 1].set_title('Cycle-Recon2 Loss')
        axs[0, 2].plot(pd.DataFrame(G_total_losses))
        axs[0, 2].set_title('G_Total Loss')
        axs[1, 2].plot(pd.DataFrame(D1_losses))
        axs[1, 2].set_title('D1 Loss')
        axs[2, 2].plot(pd.DataFrame(D2_losses))
        axs[2, 2].set_title('D2 Loss')
        axs[3, 2].plot(pd.DataFrame(D_total_losses))
        axs[3, 2].set_title('D_Total Loss')

        plt.tight_layout()
        plt.savefig(os.path.join(self.result_path, 'evaluation.png'))
        plt.show()

    def compute_kl(self, mu):
        mu_2 = torch.pow(mu, 2)
        loss = torch.mean(mu_2)
        return loss
