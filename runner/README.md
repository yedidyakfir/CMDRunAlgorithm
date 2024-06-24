# CMD Run Algorithm
This package is meant to allow you to run any configuration of your code from the cli without needing to specially configure the code for it.
Each class and every function you create can be run from the cli, you can configure the parameters for each run and add default or special rules.

## Examples
- To set a new type for a parameters use _type ending `python -m run class_name --parameter_type TypeClassName`
- For nested parameters, we allow to configure with nested naming  `python -m run class_name --model.optimizer.lr=0.01`
- If a new type isn't a class you created or doesn't inherit from the base class specified as the parameter type and needed special configuration. 
  You can use rules to set them `python -m run class_name --optimizer_type Adam --rule optimizer.lr=0.01`
