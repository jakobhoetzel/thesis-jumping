## Code for Jumping

### How to use this repo
There is nothing to compile here. This only provides a header file for an environment definition. Read the instruction of raisimGym. 

### Dependencies
- raisimgym

### Run

1. Compile raisimgym: ```python setup develop```
2. run runner.py of the task: ```cd raisimGymTorch/env/envs/minicheetah_locomotion && python ./runner.py```

### Test policy

1. Compile raisimgym: ```python setup develop```
2. run tester.py of the task with policy if folder name set in file: ``` python raisimGymTorch/env/envs/minicheetah_locomotion/tester.py```
3. run tester.py of the task with policy if folder name not set in file: ``` python raisimGymTorch/env/envs/minicheetah_locomotion/tester.py --weight data/roughTerrain/FOLDER_NAME/policy_XXX```

### Debugging
1. Compile raisimgym with debug symbols: ```python setup develop --Debug```. This compiles <YOUR_APP_NAME>_debug_app
2. Run it with Valgrind. I strongly recommend using Clion for debugging

### Explanation of different branches needed for jumping
1. Switch_Policies: Combines running and jumping. The selection ins done in NetworkSelector.py
2. JumpingTraining: Trains the jumping over the hurdle.
3. RetrainRunner: Trains the running using sampled initialization. Can be used for pure running training when this is switched off.
4. BaselineOneNetwork: Trains running in jumping in one network
