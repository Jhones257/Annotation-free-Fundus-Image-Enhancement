import torch
print(torch.cuda.is_available()) # Deve retornar True
device = torch.device("cuda")
x = torch.ones(1000, 1000).to(device) # Joga um dado na GPU