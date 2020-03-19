""" Set up SceneWalk Code"""
from setuptools import setup, find_packages

setup(
    name='scenewalk',
    version='1.0',
    description='SceneWalk model of dynamic scan path generation',
    author='Lisa Schwetlick',
    author_email='lisa.schwetlick@uni-potsdam.de',
    packages=find_packages(),
    install_requires=['numpy',
                    'pandas',
                    'seaborn',
                    'scipy',
                    'opencv_python',
                    #'arviz',
                    'pytest',
                    'multiprocess',
                    'gitpython',
                    'pyyaml',
                    'pydream']
                    #'PyDREAM @ git+ssh://git@github.com/lschwetlick/PyDREAM.git@lisa#egg=PyDREAM-0'],  #external packages as dependencies
)
