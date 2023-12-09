
from setuptools import setup, find_packages


setup(
    name="gsap-ner",
    version="1.0",
    packages=find_packages(include=['train/train.py', '*.py']),
    install_requires=[
        'datasets==2.15.0',
        'transformers==4.35.2',
        'evaluate==0.4.1',
        'seqeval==1.2.2',
    ],
    entry_points={
            'console_scripts': ['train=train.train:main']
    },

)
