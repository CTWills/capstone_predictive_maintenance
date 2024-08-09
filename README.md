# Overview

Using data from the [Universite of Irvine California](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) multiple logistic regression models are created to predict and classify what type of failure can occur based on the machines health metrics. A hypothesis test is also performed to see if there is a difference between the rotational speed of working machines and failed machines.

# Data
There are 10000 rows and 14 columns of data. Each row represents a unique machine
#### Features
* Air temperature (kelvin)
    * The ambient temperature 
* Process temperature (kelvin)
    * Interal temperature of the machine
* Rotational speed (rpm)
* Torque (Nm)
* Tool wear (min)
    * How long the machine was running for
* Product Quality (Low, Medium, High)

#### Targets
All the targets are binary values, 0 if working and 1 if failed
* Machine failure
* TWF (Tool wear failure)
* HDF (Heat dissapation failure)
* PWF (Power failure)
* OSF (Overstrain failure)
* RNF (Random failure)

# Hypothesis Test

### Null: The average rotational speed of failed machines is the same as working machines
### H0: u1 = u2
### Alt: The average rotational speed of failed machines is not the same as working machines
### Ha: u1 != u2 (two-tailed)
### Alpha: 0.05

#### Results
T-statistic= -2.086776240544015 <br>
pvalue= 0.03764747644388529 <br> 
df= 342.4998844387028 <br>

The two-tailed p-value of 0.038 is less than the alpha of 0.05. There is sufficient statistical evidence to reject the null hypothesis.
The negative test statistic indicates the average rmp of failed machines is slower than the working machines.

# Logistic Regression Models

![screenshot](images/confusion_matrix_HDF.png)

![screenshot](https://github.com/CTWills/capstone_predictive_maintenance/tree/main/images/confusion_matrix_Machine_failure.png)

![screenshot](https://github.com/CTWills/capstone_predictive_maintenance/tree/main/images/confusion_matrix_OSF.png)

![screenshot](https://github.com/CTWills/capstone_predictive_maintenance/tree/main/images/confusion_matrix_PWF.png)

![screenshot](https://github.com/CTWills/capstone_predictive_maintenance/tree/main/images/confusion_matrix_RNF.png)

![screenshot](https://github.com/CTWills/capstone_predictive_maintenance/tree/main/images/confusion_matrix_TWF.png)