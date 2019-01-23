# PSZT-Wine_Classification
Neural network written in Python telling if wine is good or not.

<h2>Setup</h2>
<h4>Install pipenv</h4>

    pip install --user pipenv
<p>WARNING!: If you are using Windows, remember to add installation directory to your PATH variable
<h4>Create Virtual Environment</h4>

    pipenv install
  
<h4>Running aps</h4>

    pipenv run python <file>

<h4>Adding dependencies</h4>

    pipenv install <name_of_dependency>

<h4>Cupy</h4>
<p>If you don't have Nvidia graphics card to use Cupy, then in file NetworkCupy.py change</p>
<p>import cupy as cp -> import numpy as cp</p>

<h4>Running in terminal</h4>
<p>First activate pipenv shell by typing</p>

    pipenv shell
        
<p>Then simply type</p>

    python main.py
