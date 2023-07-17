import torch
from madrona_learn import profile
torch.manual_seed(0)

a = torch.randn(100000, 10000, device='cuda:0')
b = torch.randn(100000, 10000, device='cuda:0')

with profile('Add', gpu=True):
    c = a + b

torch.cuda.synchronize()

profile.commit()
profile.report()
profile.clear()

with profile('Add', gpu=True):
    for i in range(1000):
        a = a + b

torch.cuda.synchronize()

profile.commit()
profile.report()
profile.clear()

for i in range(1000):
    with profile('Add', gpu=True):
        a = a + b

torch.cuda.synchronize()

profile.commit()
profile.report()
profile.reset()

for i in range(1000):
    with profile('Add', gpu=True):
        a = a + b

torch.cuda.synchronize()

profile.commit()
profile.report()
profile.clear()
