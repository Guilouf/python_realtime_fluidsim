# Python "realtime" fluidsim

Based on the Jos Stam paper https://www.researchgate.net/publication/2560062_Real-Time_Fluid_Dynamics_for_Games
and the mike ash vulgarization https://mikeash.com/pyblog/fluid-simulation-for-dummies.html

## Install

`pip install -r requirements.txt`

# Run

- `python3 fluid_sim.py` to run matplotlib animation
- `python3 to_gif.py` to export as an animated GIF


## Issues

- For pycharm users, enable 
    ```       
    import matplotlib
    matplotlib.use('Qt5Agg')  # or 'TkAgg' or whatever works
    ```
    to disable the matplotlib view pycharm feature
