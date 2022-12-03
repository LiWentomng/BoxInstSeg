import os
import glob
import torch
import shutil
os.environ['LD_LIBRARY_PATH'] = '/home/anaconda3/envs/lib'
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))
extensions_dir = os.path.join(this_dir, "src")

main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
source_cpu = glob.glob(os.path.join(extensions_dir, "*", "*.cpp"))
source_cuda = glob.glob(os.path.join(extensions_dir, "*", "*.cu"))

if torch.cuda.is_available():
    if 'LD_LIBRARY_PATH' not in os.environ:
        raise Exception('LD_LIBRARY_PATH is not set.')
    cuda_lib_path = os.environ['LD_LIBRARY_PATH'].split(':')
    sources = source_cpu + source_cuda + main_file
else:
    raise Exception('This implementation is only avaliable for CUDA devices.')


setup(
    name='tree_filter',
    version="0.1",
    description="learnable tree filter for pytorch",
    ext_modules=[
        CUDAExtension(
            name='tree_filter_cuda',
            include_dirs=[extensions_dir],
            sources=sources,
            library_dirs=cuda_lib_path,
            extra_compile_args={'cxx':['-O3'],
                                'nvcc':['-O3']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

