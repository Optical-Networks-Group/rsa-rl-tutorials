
## RSA-RL Tutorials
This repository contains examples of training & evaluation with *RSA-RL*. 

## Structure
- src: *.py files
- tutorials: *.ipynb notebooks


## Example
Please move to src directory.   

1. Evaluate KSP-FF Agent
```
$ python ksp-agent.py [-sa ff] [-db rsa-rl.db] [--overwrite] --save
```
When you use _Logger_, 
you must not use the experimental name that is already in the database. 
If you want to overwrite the already experimental information, 
then add option _overwrite_. 
After execution, _rsa-rl.db_ is generated. 
This database includes logs that include which path and frequency slots were selected by the KSP-FF Agent. To visualize it, execute
```
$ rsa-rl-visualizer rsa-rl.db
```

