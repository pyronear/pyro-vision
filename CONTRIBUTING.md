# Contributing to PyroNear

Everything you need to know to contribute efficiently to the project.



## Codebase structure

- [pyronear](https://github.com/frgfm/PyroNear/blob/master/pyronear) - The actual PyroNear library
- [references](https://github.com/frgfm/PyroNear/blob/master/references) - Scripts to reproduce performances
- [test](https://github.com/frgfm/PyroNear/blob/master/test) - Python unit tests



## Continuous Integration

This project uses the following integrations to ensure proper codebase maintenance:

- [CircleCI](https://circleci.com/) - run jobs for package build and coverage
- [Codacy](https://www.codacy.com/) - analyzes commits for code quality
- [Codecov](https://codecov.io/) - reports back coverage results

As a contributor, you will only have to ensure coverage of your code by adding appropriate unit testing of your code.



## Issues

Use Github [issues](https://github.com/frgfm/PyroNear/issues) for feature requests, or bug reporting. When doing so, use issue templates whenever possible and provide enough information for other contributors to jump in.



## Code contribution

In order to contribute to  project, we will first **set up the development environment**, then describe the **contributing workflow**.

* [Project Setup](project-setup)

    _How to set up a forked project and install its dependencies in a well-encapsulated development environment_
    1. [Create a virtual environment](#create-a-virtual-environment)
    2. [Fork the project](#fork-the-repository)
* [Contributing workflow](contributing-workflow)

   _How to pull remote changes/new contributions and push your contributions to the original project_

* [Code & commit guidelines](#commits)

### Project Setup
---
In order to enable every one to fluently contribute to the project, we are going
to set up the project properly following some steps:
1. **Create a virtual environment** to avoid collision with our OS and other projects
2. **Fork the project** to be able to start working on a local copy of the project

#### 1. Create a virtual environment
We are going to create an python3.6 environment with dedicated to Pyro project. We'll use [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/).

Please open a terminal and follow the instructions.
```shell
# install package
pip install virtualenvwrapper

# add at the end of your .bashrc
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/Devel
source /usr/local/bin/virtualenvwrapper.sh

# list available virtual environments
workon

# create new environment dubbed "pyro36" using python 3.6
mkvirtualenv -p $(which python3.6) pyro36

# activate pyro36 environment
workon pyro36

# deactivate the current environment
deactivate

# delete virtual environment (only do it if needed)
rmvirtualenv pyro36
```

#### 2. Fork the repository
We are going to get a local copy of the remote project (_fork_) and set remotes so we stay up to date to recent contributions.

1. Create a fork by clicking on the **fork button** on the current repository page
2. Clone _your_ fork locally.
```shell
# change directory to one for the project
cd /path/to/local/pyronear/project/

# clone your fork. replace YOUR_USERNAME accordingly
git clone https://github.com/YOUR_USERNAME/PyroNear.git
```

3. Set remotes to original project and merge new contributions onto master.
```shell
# add the original repository as remote repository called "upstream"
git remote add upstream https://github.com/frgfm/PyroNear.git

# verify repository has been correctly added
git remote -v

# fetch all changes from the upstream repository
git fetch upstream

# switch to the master branch of your fork
git checkout master

# merge changes from the upstream repository into your fork
git merge upstream/master
```

4. install the project dependencies
```shell
# install dependencies
pip install -r requirements.txt

# install current project in editable mode,
# so local changes will be reflected locally (ie:at import)
pip install -e .
```


### Commits

- **Code**: ensure to provide docstrings to your Python code. In doing so, please follow [Google-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) so it can ease the process of documentation later.
- **Commit message**: please follow [Udacity guide](http://udacity.github.io/git-styleguide/)
