from setuptools import setup, find_packages
from codecs import open
from os import path

setup(
    name='coffeehouse_dltc',
    version='1.0.0',
    description='Deep Learning Text Classification Engine',
    url='https://github.com/Intellivoid/CoffeeHouse-DLTC',

    # Author details
    author='Zi Xing Narrakas',
    author_email='netkas@intellivoid.info',

    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Internal/Alpha',
        'Topic :: Text Processing',
        'Programming Language :: Python :: 3',
    ],

    # What does your project relate to?
    keywords='multi-label classification nlp neural networks deep learning',

    packages=find_packages(exclude=['tests']),

    install_requires=[
        'nltk~=3.4',
        'numpy~=1.17',
        'scipy~=1.3.1',
        'gensim~=3.8.0',
        'scikit-learn~=0.21.3',
        'keras~=2.2.5',
        'h5py~=2.9',
        'tensorflow~=1.14.0',
        'six'
    ],

    entry_points='''
        [console_scripts]
        magpie=magpie.linear_classifier.cli:cli
    ''',
)
