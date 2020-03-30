.. _getting_started:

================
Getting Started
================

1. activate your virtual environment and/or make one::

        cd ~/Documents/virtual_envs
        python3 -m venv sw_env
        source sw_env/bin/activate

2. install scenewalk package::

        cd ~/Scenewalk
        pip3 install -e .

3. install the venv as a kernel for notebooks::
        
        pip3 install ipykernel
        python3 -m ipykernel install --name sw_env

4. add the location of your Data to the config file. Simply go to the `config.yml` file and change the paths and/or dataset names!

5. use it like any other package!

