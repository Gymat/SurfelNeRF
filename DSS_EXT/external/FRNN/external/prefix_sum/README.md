# Parallel Prefix Sum (Scan) with CUDA 

## Pytorch Usage Note
### Installation
```
python setup.py install
```
### Usage
```
from prefix_sum import prefix_sum_cpu, prefix_sum_cuda
# assuming input is a torch.cuda.IntTensor, num_elements is an integer
# allocate output_array on cuda
# e.g. output = torch.zeros((num_elements,), dtype=torch.int, device=torch.device('cuda'))
prefix_sum_cuda(input, num_elements, output)

# similarly for the CPU version
# except that both input and output are torch.IntTensor now
prefix_sum_cpu(input, num_elements, output)
```

## Original README

My implementation of parallel exclusive scan in CUDA, following [this NVIDIA paper](http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf).

>Parallel prefix sum, also known as parallel Scan, is a useful building block for many
parallel algorithms including sorting and building data structures. In this document
we introduce Scan and describe step-by-step how it can be implemented efficiently
in NVIDIA CUDA. We start with a basic naïve algorithm and proceed through
more advanced techniques to obtain best performance. We then explain how to
scan arrays of arbitrary size that cannot be processed with a single block of threads. 

This implementation can handle very large arbitrary length vectors thanks to the [recursively defined scan function](https://github.com/mattdean1/cuda/blob/master/parallel-scan/scan.cu#L105).

Performance is increased with a memory-bank conflict avoidance optimization (BCAO).

---

See the [timings](https://github.com/mattdean1/cuda/blob/master/parallel-scan/Submission.cu#L616) for a performance comparison between:
  1. Sequential scan run on the CPU
  2. Parallel scan run on the GPU
  3. Parallel scan with BCAO
  
For a vector of 10 million entries:

	  CPU      : 20749 ms
	  GPU      : 7.860768 ms
	  GPU BCAO : 4.304064 ms
    
    Intel Core i5-4670k @ 3.4GHz, NVIDIA GeForce GTX 760
