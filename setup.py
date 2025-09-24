from setuptools import setup, find_packages

setup(
    name='ebm_lib',
    version='0.1.0',
    description='A modular research library for Energy-Based Models',
    author='Jules',
    packages=find_packages(include=['ebm_lib', 'ebm_lib.*']),
    install_requires=[
        'torch',
        'pytorch-lightning',
        'albumentations',
        'numpy',
        'scipy',
        'torchvision',
    ],
    python_requires='>=3.8',
)
