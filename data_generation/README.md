# Data Generation

We use the package 'generate gravitational-wave data' by Timothy Gebhard.
The folder 'ggwd' contains a clone of the package from https://github.com/timothygebhard/ggwd.

To recreate the testing and training data sets described in [Xia et al., 2020](https://arxiv.org/pdf/2011.04418.pdf), we have included the following additional scripts under `ggwd`:
 - generate_testing_data.py
 - generate_training_data.py

To run the scripts, type one of the following commands into the console,
```
python generate_testing_data.py -n N
python generate_training_data.py -n N
```
It is important to note that on computers with Python 2 and Python 3 installed, it may be necessary to use `python3` instead of `python` in the previous commands.

This will generate the `N`th testing or training data set presented in Table 1 of [Xia et al., 2020](https://arxiv.org/pdf/2011.04418.pdf).
