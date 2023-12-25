from setuptools import setup, find_packages

setup(
    name='SeisHandler',
    version='0.2.0',
    author='Wang Jingxi',
    author_email='1531051129@qq.com',  # 作者邮箱
    description='A utility for handling seismic array data files.',  # 项目简介
    long_description=open('README.md').read(),  # 通常是你项目的README
    long_description_content_type='text/markdown',  # README文件的格式
    url='https://github.com/wangkingh/SeisHandler',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.0.3',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Natural Language :: English', 
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Linux',
    ],
    python_requires='>=3.6',
)