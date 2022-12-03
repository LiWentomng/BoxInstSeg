from setuptools import find_packages, setup
import os
import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):
    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


if __name__ == '__main__':
    setup(
        name='boxinst_plugin',
        version='0.0.1',
        description=('boxinst_plugin'),
        long_description='boxinst_plugin',
        long_description_content_type='text/markdown',
        author='zhanggefan',
        author_email='lizaozhouke@sjtu.edu.cn',
        keywords='computer vision, 3D object detection',
        packages=find_packages(),
        include_package_data=True,
        license='Apache License 2.0',
        ext_modules=[
            make_cuda_ext(
                name='pairwise_ext',
                module='.',
                sources=[
                    'csrc/pairwise/bind.cpp',
                    'csrc/pairwise/pairwise.cu'
                ])],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
