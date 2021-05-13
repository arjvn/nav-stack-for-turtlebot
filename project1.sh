source ~/ws/devel/setup.bash
chmod +x ~/ws/src/pkg/scripts/*.py
# all environment variables MUST NOT contain spaces
export GOALS="3.0,-4.0|-1.0,-5.0|-3.5,-8.5"
export X_POS="-2.0"
export Y_POS="-0.5"
export WORLD="world2020"
export CELL_SIZE="0.1"
export MIN_POS="-8.0,-13.0"
export MAX_POS="7.0,3.0"

roslaunch pkg project1.launch

# ====== TURTLEBOT3_WORLD =========
#export GOALS="0.5,0.5|1.5,0.5|-1.5,-1.5|-2.0,-0.5"
#export X_POS="-2.0"
#export Y_POS="-0.5"
#export WORLD="turtlebot3_world"
#export CELL_SIZE="0.1"
#export MIN_POS="-4.0,-4.0"
#export MAX_POS="4.0,4.0"

# ====== WORLD1 =========
#export GOALS="-4.0,-6.0|-2.5,-1.0|-9.0,1.5"
#export X_POS="-8.9"
#export Y_POS="-8.7"
#export WORLD="world1"
#export CELL_SIZE="0.1"
#export MIN_POS="-11.0,-11.0"
#export MAX_POS="11.0,11.0"

# ====== WORLD2 =========
#export GOALS="2.0,0.0|3.0,-4.0|-4.0,-6.0"
#export X_POS="-2.0"
#export Y_POS="-0.5"
#export WORLD="world2"
#export CELL_SIZE="0.1"
#export MIN_POS="-7.0,-11.0"
#export MAX_POS="7.0,4.0"


# ====== WORLD3 =========
#export GOALS="3.0,-4.0|-1.0,-5.0|-0.5,-9.5"
#export X_POS="-2.0"
#export Y_POS="-0.5"
#export WORLD="world3"
#export CELL_SIZE="0.1"
#export MIN_POS="-6.0,-12.0"
#export MAX_POS="6.0,6.0"
