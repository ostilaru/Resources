from setuptools import setup, find_packages

setup(
    name='FastXC',
    version='0.1.0',
    description='A package call cuda execuatable file to do fast cross-correlation',
    author='Wang Jingxi',
    author_email='1531051129@qq.com',
    url='https://github.com/wangkingh',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    package_data={
        'FastXC': ['FastXC/*.ini'],
    },
    include_package_data=True,
)

