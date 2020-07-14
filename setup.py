from setuptools import setup, find_packages

setup(
    name='coffeehouse_dltc',
    version='1.0.1',
    description='Deep Learning Text Classification Engine',
    url='https://github.com/Intellivoid/CoffeeHouse-DLTC',
    author='Zi Xing Narrakas',
    author_email='netkas@intellivoid.info',
    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',
        'Topic :: Text Processing',
        'Programming Language :: Python :: 3',
    ],
    keywords='multi-label classification nlp neural networks deep learning',
    packages=find_packages(),
    install_requires=[
        'nltk',
        'numpy~=1.17',
        'scipy~=1.3.1',
        'gensim~=3.8.0',
        'scikit-learn==0.22.0',
        'keras~=2.2.5',
        'h5py~=2.9',
        'tensorflow~=1.14.0',
        'six',
        'cython'
    ],
    entry_points='''
        [console_scripts]
        coffeehouse_dltc=coffeehouse_dltc.linear_classifier.cli:cli
    ''',
)
