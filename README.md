# Assigment 4: Finding Materials with High Hardness

Perform Bayesian optimization with featurization and a contextual variable to
maximize hardness as a function of material composition.

## The Assignment

Some data representations cannot be directly used for optimization due to a lack of
continuity or differentiability. In such cases, it is necessary to transform the data
into a form that can be more easily understood by machine learning models. This
process is called featurization and is very common in the materials, chemistry, and
biological informatics fields. In this assignment, you will use Ax to help select a
material composition with a high hardness based on a list of potential candidates.
This differs from previous assignments in that candidate data is discrete and will be
fed to the optimization model manually.

Thankfully you won't be starting from scratch! An intern spent some time looking
through the literature and found hardness data as a function of composition
and stored it in a CSV file called `train.csv`. Unfortunately, much of the data was
captured using different hardness testers, so the applied loads used in the literature
tests are differnet. Your machine is set to a load of 0.98 kgf, which wasn't used in
any of the literature data. You suspect that by including the load as a feature, the
model will be able to account for this discrepancy. You have devised a list of
potential candidate materials and stored them in a CSV file called `candidate.csv`.
These will be used the candidates you select from at each optimization iteration.

Your goal is to use Honegumi and your knowledge of the Ax API to develop an
optimization script to help find a composition that maximizes the yield strength
of a developed alloy. Your experimental budget is limited to 25 experiments. A
synthetic objective function has been provided that will serve as a proxy for real
experimental measurements.

### **TASK A:** Featurize the data using the featurize_data function.

Featurize the the training set and the candidate set using the provided function.
Store the results in the variables X_train, y_train, and X_candidates. Note that as
the candidate data hasn't be evaluated, the harndness values are all set to 0.

The `featurize_data()` function takes in a dataframe with material compositions
and properties and returns a dataframe with the features, `X`, and a dataframe
with the property values `y`.

### **TASK B:** Use Honegumi to set up the optimization problem and attach the training data.

Honegumi will help with the general framework, but you will need to make several
changes to the typical structure to allow discrete candidates to be passed and
evaluated. We provide a few helpful tips below for making these modifications.

1. As solutions are drawn from a discrete list, a typical Sobol generation strategy
at the start will not work. You will need to use a custom generation strategy that
immediately starts with a GP model - we reccommend `BOTORCH_MODULAR` for this.

2. The parameter names are fixed to the columns of the featurized DataFrame. We will
assume that all of the generated candidates are feasible and can set arbitratiy large
bounds for the ranges on these parameters to avoid Ax throwing an error.

3. The training data will need to be attached to the optimization problem using the
`attach_trial` method. This will serve as the initial training set for the optimization.

### **TASK C:** Build the optimization loop and and evaluate 25 candidates.

Now you will build an optimization loop that passes the candidate features to the
trained model, returns the acquisition function for each, and selects a candidate to
evaluate. Reminder that passing custom parameter evaluations to the model requires
one to create an ObservationFeatures object. For evaluations, you will use the
provided `measure_hardness` function.

Some pseudocode for getting acquisition function values from manually passed data
is provided below to help you construct the manual optimization loop.

```python
ax_client.fit_model()
model = ax_client.generation_strategy.model 
obs_feat = ObservationFeatures({"feat":val})
acqf_value = model.evaluate_acquisition_function(obs_feat)
```

### **TASK D:** Report the optimal composition and associated hardness.

Now that you have completed the optimization, identify the chemical composition of the
optimial parameters parameter combination and assign it to a variable named
`optimal_composition`. Assign the optimal hardness to a variable named `max_hardness`.

### **TASK E:** Report the most important feature and its correlation for predicting hardness.

Identify the most important feature for predicting hardness and assign it to a
variable named `most_important`. Then indicate whether it is positively (1) or
negatively (-1) correlated with hardness and assign it to a variable named `correlation`.

### **TASK F:** How many materials have a hardness greater than 43?

There are often secondary constraints in an optimization problem and having several
options to choose from can be helpful. Determine how many materials have a hardness
greater than 43 and assign the result to a variable named `n_hard_materials`.


## Setup command

See `postCreateCommand` from [`devcontainer.json`](.devcontainer/devcontainer.json).

## Run command
`pytest`

## Notes
- pip's install path is not included in the PATH var by default, so without installing via `sudo -H`, pytest would be inaccessible.
