import torch
import os
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

class BuildExtension(torch.utils.cpp_extension.BuildExtension):
    def __init__(self, *args, **kwargs):
        super().__init__(use_ninja=False, *args, **kwargs)

nvcc_args = []
nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
if nvcc_flags_env != "":
    nvcc_args.extend(nvcc_flags_env.split(" "))

extra_compile_args = {}
extra_compile_args["nvcc"] = nvcc_args

setup(
  name="prefix_sum",
  author="Matt Dean, Lixin Xue",
  description="Parallel Prefix Sum on CUDA with Pytorch API",
  ext_modules=[
    CUDAExtension('prefix_sum', ['prefix_sum.cu'], extra_compile_args=extra_compile_args)
  ],
  cmdclass={"build_ext": BuildExtension},
)
