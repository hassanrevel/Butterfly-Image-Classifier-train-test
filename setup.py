from setuptools import setup, find_packages

setup(
    name= 'ButterlyImageClassifier',
    version='0.1.0',
    description='It tells the species of butterfly',
    license='MIT',
    author='A.H Revel',
    author_email= 'alihassanrevel@gmail.com',
    maintainer='A.H Revel',
    maintainer_email= 'alihassanrevel@gmail.com',
    classifiers=[
        'Development Status :: In progress',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python >= 3.7',
        'Topic :: DataScience/Deeplearning/ComputerVision',
    ],

    # install_requires =[
    #     'torch',
    #     'matplotlib',
    #
    # ],

    packages= find_packages()
)

