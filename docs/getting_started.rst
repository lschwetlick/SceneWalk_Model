.. _getting_started:

================
Getting Started
================

1. download the code repo from https://github.com/lschwetlick/SceneWalk_Model

2. activate your virtual environment and/or make one::

        python3 -m venv sw_env
        source sw_env/bin/activate

3. install scenewalk package. The `-e` flag is for installing in "developer mode, so that changes made to the code are loaded. Without the -e flag, Changes will not affect the installed module."::

        cd ~/Scenewalk
        pip3 install -e .

4. (optional) install the venv as a kernel for notebooks::
        
        pip3 install ipykernel
        python3 -m ipykernel install --name sw_env

5. (optional) add a `config.yml` file into the scenewalk folder. This yml file should follow the same structure ar `sample_config.yml` and gives the loading module the information about where to look for the data. Please refer to the section "Data Requirements" for more information about how data is used.  

6. use the package!

