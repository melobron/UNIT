import torch

mu = torch.ones((1, 4, 4))

z = torch.zeros_like(mu).normal_(mean=0, std=1).to(mu.device)
print(z.device)

