
from setuptools import setup, find_packages


setup(
    name="gsap",
    version="1.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    #, "gsap.training": "src/gsap/training"}, #, "gsap.data": "data"},
    install_requires=[
        'datasets==2.15.0',
        'transformers==4.35.2',
        'evaluate==0.4.1',
        'seqeval==1.2.2',
        'accelerate==0.25.0',
    ],
    # find your version: 'https://download.pytorch.org/whl/'
    extras_require={
        #"TORCH": ["torch@https://download.pytorch.org/whl/cu117/torch-2.0.1%2Bcu117-cp310-cp310-linux_x86_64.whl"],
        "TORCH": ["torch@https://download.pytorch.org/whl/cu117/torch-1.13.1%2Bcu117-cp310-cp310-linux_x86_64.whl"],
    },
    dependency_links=[
    ],
    entry_points={
            'console_scripts': ['gsap-ner-train=gsap.training.train:main']
    },
)
