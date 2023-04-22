# neuralODE-panda
Using NeuralODE to Learn Panda Arm Dynamics


To install dependencies, run
```
./install.sh
```

Then, to run the demo, run
```
python demo.py
```
The demo evaluates two types of standard NNs and NeuralODEs on validation data and performs MPPI control to move a block to a goal position which has an obstacle blocking the path.
