# cmd-run-algorithm
This package allow you to run classes function from command line. This can help with running multiple configuration for algorithms without changing the code or making any special adapters

- You can use list type by setting the parameter multiple times `python run class --param 1 --param 23` 

Notes:
- If you use typing types, you'll have to specify the type using _type param. This feature is in development.
`python run class --param_type int --param 1`
- List param can only by used by primitive types currently


## Future features:
- Create a nested structure to analyze the correct data type. Supporting nested types (Optional, Union, List etc)
- Allow user to create a connection from inner parameter that was not directly created.
<br>for example, if I need a.b.c->model and I just created a, I dont want to create a special creator for this. make a way to take the inside of a and pass it to the creator
- Allow setting consts (for example when we want set dtype to torch.float16, add additional param for this like --param_name-const)
- how to manipulate the class, like setting the start point?
- add settings and default config

### cli features
- always run a specific function and a specific class
- run multiple functions one after the other
- test the cli

- create a module script to run code, (the script should pick a function or a class to run using python -m) 
- create docs for usage in packge