""" Set up SceneWalk Code"""
from setuptools import setup, find_packages

setup(
    name='scenewalk',
    version='1.0',
    description='SceneWalk model of dynamic scan path generation',
    author='Lisa Schwetlick',
    author_email='lisa.schwetlick@uni-potsdam.de',
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'seaborn', 'scipy', 'pymc3', 'pytest', 'multiprocess', 'gitpython'], #external packages as dependencies
)
