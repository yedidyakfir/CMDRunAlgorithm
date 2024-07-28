# cmd-run-algorithm
This package allow you to run classes function from command line. This can help with running multiple configuration for algorithms without changing the code or making any special adapters

- You can use list type by setting the parameter multiple times `python run class --param 1 --param 23` 

Notes:
- If you use typing types, you'll have to specify the type using _type param. This feature is in development.
`python run class --param_type int --param 1`
- List param can only by used by primitive types currently


## Future features:
- allow multiple modules to serach in
- we need a way to pick different option for maniuplation parameters with functions, maybe creators?
- Create a nested structure to analyze the correct data type. Supporting nested types (Optional, Union, List etc)
- how to manipulate the class, like setting the start point?
- add option to request parameters for additional function (like user who wants to activate egl.train) inside another function

### cli features
- always run a specific function and a specific class
- run multiple functions one after the other
- test the cli

- create a module script to run code, (the script should pick a function or a class to run using python -m) 
- create docs for usage in packge
- The config is in a list and not hierarchy, is there a way to convert it?

### Tests
- add test for run function for use_config option