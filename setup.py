from setuptools import setup, find_packages

VERSION = '0.0.2'
DESCRIPTION = 'Create neural networks'
LONG_DESCRIPTION = 'A package that allows you to easily create and run neural networks.'

# Setting up
setup(
    name="deepneuron",
    version=VERSION,
    author="Vignesh Reddy",
    author_email="<vignesh77reddy@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['numpy'],
    keywords=['python', 'neural', 'network', 'neural network', 'deep learning', 'deep', 'learning'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        'License :: OSI Approved :: MIT License',
    ]
)
