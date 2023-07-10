import torch
from prefix_sum import prefix_sum_cuda, prefix_sum_cpu
import numpy as np

def test(num_pcs, max_num_grids):
  gpu = torch.device("cuda:0")
  cpu = torch.device("cpu")
  grid_cnt = torch.randint(low=0, high=1000, size=(num_pcs, max_num_grids), dtype=torch.int, device=cpu)
  grid_off = torch.full(size=grid_cnt.shape, fill_value=-1, dtype=torch.int, device=cpu)

  # grid_cnt_cuda = torch.randint(low=0, high=1000, size=(num_pcs, max_num_grids), dtype=torch.int, device=gpu)
  grid_cnt_cuda = grid_cnt.cuda()
  grid_off_cuda = torch.full(size=grid_cnt.shape, fill_value=-1, dtype=torch.int, device=gpu)

  for i in range(num_pcs):
    num_grids = np.random.randint(low=0, high=max_num_grids)
    prefix_sum_cpu(grid_cnt[i], num_grids, grid_off[i])
    prefix_sum_cuda(grid_cnt_cuda[i], num_grids, grid_off_cuda[i])
    print(grid_off[i, :20])
    print(grid_off_cuda[i, :20])

  print(torch.allclose(grid_off, grid_off_cuda.cpu()))


if __name__ == "__main__":
  test(100, 1000000)
