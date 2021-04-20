# Data Generation

We use the package 'generate gravitational-wave data' by Timothy Gebhard.
The folder 'ggwd' contains a clone of the package from https://github.com/timothygebhard/ggwd.

We have included the following original scripts to generate the testing and training datasets as described in https://arxiv.org/pdf/2011.04418.pdf.
 - generate_testing_data.py
 - generate_training_data.py
To run the scripts, type one of the following commands into the console,
```python
python3 generate_testing_data.py -n N
python3 generate_training_data.py -n N
```
This will generate the `N`th dataset seen in the table in (Xia et al, 2011)