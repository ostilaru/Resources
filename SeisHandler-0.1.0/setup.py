from setuptools import setup, find_packages

setup(
    name="SeisHandler",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    author='Wang Kinghei and ChatGPT',
    author_email='wkh16@mail.ustc.edu.cn',
    long_description=open('README.md').read(),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    install_requires=[
        "pandas",
    ],
)
