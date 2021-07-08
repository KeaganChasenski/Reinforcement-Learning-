# Assignment 2 - RL]
    #ValueIteration.py

# Keagan Chasenski
# CHSKEA001
# 28th May 2021

#Returns
    #[Records]: shown as plt animation
    #[Parameter] : all the params used to run the ValueIteration algorithm
    #[Mines] : location of all the mines
    #[Iteration] : number of iterations to converge

import numpy as np
import matplotlib.pyplot as plt
import sys
import random
from Animate import generateAnimat

# Hyper parameters
theta = 0.005

def setArguments(argv):
    # Arguments 
    global width 
    global height 
    global x_start
    global y_start
    global x_end
    global y_end 
    global k 
    global g

    # Correct at least width and heigh are specfied in the run
    if len(argv) < 2:
        print("Not enough arguments, Please re run providing the size of the grid")
        sys.exit()
    else:

        # Set width, height
        width = int(argv[1])
        height = int(argv[2])

        # if start specified, set start coordinates
        if "-start" in argv:
            start = argv.index("-start")
            x_start = int(argv[start + 1])
            y_start = int(argv[start + 2])
        
        else:
            x_start = random.randint(0,width-1)
            y_start = random.randint(0,height-1)
        # if end specified, set start coordinates
        if "-end" in argv:
            end = argv.index("-end")
            x_end = int(argv[end + 1])
            y_end = int(argv[end + 2])
        # Otherwise randomly generate these
        else:
            x_end = random.randint(0,width-1)
            y_end = random.randint(0,height-1)
            # ensure that the end is not randomly assigned to be the start 
            while x_start == x_end and y_start == y_end:
                x_end = random.randint(0,width-1)
                y_end = random.randint(0,height-1)
        # if number of landmines specified, then set k = specified vale
        if "-k" in argv:
            num = argv.index("-k")
            k = int(argv[num + 1])

        # Default 3 mines
        else:
            k = 3 

        # Set the discount value
        if "-gamma" in argv:
            gamma = argv.index("-gamma")
            g = float(argv[gamma + 1])
        # Default 0.9 seems to work well
        else:
            g = 0.9 

def printArguments():
    # Simple function to print all the parameters to the terminal
    print("Width:" , width)
    print("Height:" , height)
    print("Start state:" , start_state)
    print("End state:" , end_state)
    print("Landmines:" , k)
    print("Gamma :" , g)

def createGrid():

    # Creates a grid of all states of height X width
    all_states = []
    for i in range(height):
        for j in range(width):
            all_states.append((i,j))
            
    return all_states 

def addMines():
    # function to create an array of landmines 
    # returns array
    # ensures not at start/ end coordinates or that duplicate mines on same location

    # mines array
    mines = []

    # Loop for the number of mines
    for i in range(k):

        # Generate random locations
        x = random.randint(0,width-1)
        y = random.randint(0,height-1)

        # Prevents landmine from being placed on start position
        while (x == x_start and y == y_start):
            x = random.randint(0,width-1)
            y = random.randint(0,height-1)
        # Prevents landmine from being placed on end position
        while (x == x_end and y == y_end):
            x = random.randint(0,width-1)
            y = random.randint(0,height-1)    

        # append the first mine to the array before checking (otherwise the first one will result in an index error)
        if i == 0:
            mines.append((y,x)) 
        else:
            # Loop through previous mines and ensure not on the same randomly generated location
            for m in mines:
                # Check that the x,y generated pair != a previos mine
                while x == m[1] and y == m[0]:
                    x = random.randint(0,width-1)
                    y = random.randint(0,height-1)
                # Prevents landmine from being placed on start position
                while (x == x_start and y == y_start):
                    x = random.randint(0,width-1)
                    y = random.randint(0,height-1)
                # Prevents landmine from being placed on end position
                while (x == x_end and y == y_end):
                    x = random.randint(0,width-1)
                    y = random.randint(0,height-1) 

            # append to coord to the array
            mines.append((y,x))
    return mines

def rewards(all_states, mines):
    # Function adds reward values to all states
    # Returns rewards dictionary for easy indexing
    # -1 for landmines
    # 1 for end state
    # 0 otherwise
    rewards = {}

    # Sets end points and all other states except mines
    for i in all_states:

        # Reward for end point
        if i == (y_end,x_end):
            rewards[i] = 1
        # Reward for all other states
        else:
            rewards[i] = 0

    # Sets the reward for mines to be -1 to prevent action from going there
    for i in all_states:
        for j in mines:
            if i == j:
                rewards[i] = -1

    return rewards 

def actions(all_states, mines):
    actions = {}
    # exclude all land mines from possible actions as we can never go there
    # check that a move is possible add if so add the action
    # Possible actions {"Up", "Down", "Left", "Right"}
    # Returns actions dictionary

    # Loop through all states
    for s in all_states:
        actions[s] = []
        # If top row, can't have an up action
        if s[0] != 0:
            actions[s].append('U')
        # If bottom row, can't have a down action
        if s[0] != height-1:
            actions[s].append('D')
        # If right column, can't have a right action
        if s[1] != width-1:
            actions[s].append('R')
        # If left column, can't have a left action
        if s[1] != 0:
            actions[s].append('L')
        # If landmine, not actions at all
        for i in mines:
            if s == i:
                del actions[s]

    return actions

def initalPolicy(actions):
    # Function to randomly choose an intial policy from the list of possible actions for each state
    # returns a dict of which action to first take at each state

    policy={}
    # for every action (excludes mines and end state (terminal) )
    for a in actions:
        # choose a random policy
        policy[a] = np.random.choice(actions[a])
  
    return policy

def initalValue(all_states):
    # Function assigns the value of each state depending on if it is a mine or the end state
    # -1 for every state (to ensure optimal policy)
    # -10 for landmines to avoid them
    # 100 for end state

    # get the possible actions list from the action function
    actions_list = actions(all_states, mines)

    # Empty dictionary of Values
    V={}

    # Loop for every state
    for s in all_states:

        # Assign -1 as a value if there is a possible action at the state (ie not terminal)
        if s in actions_list.keys():
            V[s] = -1
        # end state = 100 value
        if s[0] == y_end and s[1] == x_end:
            V[s]=100
        # for every state, check if in the mine array, if is assign -10
        for m in mines:
            if s == m:
                V[s] = -10

    return V

def valueIteration(all_states, policy, actions, V, rewards_list):
    # Impletment the value iteration algorithm

    iteration = 0 
    data = [[]]

    # Loop until threshold of theta is not changin
    while True:

        # Set the change to 0
        biggest_change = 0

        # Loop through all states
        for s in all_states: 

            # If there is a move (policy) at the current state
            if s in policy:

                # Set the Value at the state to old value
                old_v = V[s]
                # Update the new value 
                new_v = 0

                # For every possible action in the state
                for a in actions[s]:
                    # Do the action for the state by updating the next state based on the direction
                    if a == 'U':
                        nxt = [s[0]-1, s[1]]
                    if a == 'D':
                        nxt = [s[0]+1, s[1]]
                    if a == 'L':
                        nxt = [s[0], s[1]-1]
                    if a == 'R':
                        nxt = [s[0], s[1]+1]
                    # Set the next state
                    nxt = tuple(nxt)

                    # Value iteration formula

                    # If we have reached the end state -> Value is 100
                    if s[0] == y_end and s[1] == x_end:
                        v = 100
                    else:
                        # Update the value based of the ValueIteration alogirthm formula
                        # Using Belmans equation
                        v = rewards_list[s] + (g * V[nxt])

                    #If it is the best action, keep it 
                    if v > new_v: 
                        new_v = v
                        policy[s] = a
                # Append the new V to be the value at that state in the value array
                V[s] = new_v
                # Update the biggest change to be max of previous change or new change in Value 
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))  

        # Stopping condition for while loop
        # Threshold theta
        if biggest_change < theta:
            iteration += 1
            break
        
        # Update our Values list with the current state index
        state = list(V.values())
        data.append(state)
        iteration += 1 

    return iteration, data, policy

def createRecords(data):
    # function to return the records in the data structure required for animate.py
    # complex code due to tuples and dictionarys, but was too far along in the assigmnet to change :(

    # returns r_list, in the correct formatt.
    
    # data is passed from valueIteration function, delete the first record which is empty.
    del data[0]

    #black records list
    records = []

    # Since data is a 1D flat array, shape every element to be grid size and append to records array 
    for i in data:
        x = np.reshape(i, (height, width))
        records.append(x)
    
    # Convert recrods to a NumPy array (a bit unessecary)
    records = np.array(records)

    # Convert the records array to a list and then append it
    r_list = []
    for i in range(len(records)):
        r_list.append(records[i].tolist())
    
    # delete empty first value
    del r_list[0]

    return r_list

def reverseMines(mines):
    # fuction reverses the mines x,y coords due to at the start I mixed them up
    # easier to write this function than change every previous function
    # returns reversed_mines to be displayed in the plot

    reversed_mines = []
    for m in mines:
        reversed_mines.append(m[::-1])

    return reversed_mines

def optimalPolicy(policy, start_state, end_state, iteration):
    # Function returns the optimal policy for getting from start to end state
    # takes in the policy produced from ValueIteration function
    # this policy includes an action for every state, we only want from the start state and the path

    # Again because at the beginning x,y coords were entered incorrectly
    current = start_state[::-1]

    # find the start location in the policy dict
    op = [current]
    #the move or action from the location
    move = policy[current]
 
    # Loop until at the end_state 
    while current != end_state[::-1]:
        # Update the move policy
        move = policy[current]

        # Move Up in the array
        if move == 'U':
            next = (current[0]-1, current[1])
            # Add location to optimal policy
            op.append(next)
            # Update Next
            current = next

        # Move Down in the array
        if move == 'D':
            next = (current[0]+1, current[1])
            op.append(next)
            current = next

        # Move Left in the array
        if move == 'L':
            next = (current[0], current[1]-1)
            op.append(next)
            current = next

        # Move Right in the array
        if move == 'R':
            next = (current[0], current[1]+1)
            op.append(next)
            current = next

    # Reverse the element of the optimal policy to be printed in the correct formatt
    optimal_policy = []
    for p in op:
        optimal_policy.append(p[::-1])
    
    # Print Optimal Policy
    print("The optimal policy is: \n")
    print(optimal_policy, '\n')

    return optimal_policy

if __name__ == "__main__" :

    # Set the arguments for running
    setArguments(sys.argv)
    start_state = (x_start, y_start)
    end_state = (x_end, y_end)
    mines = addMines()
    # Display arguments in nice format
    printArguments()
    print("Mines:" , reverseMines(mines))

    # Create the grid
    all_states = createGrid()
    
    # Set up for ValueIteration
    rewards_list = rewards(all_states, mines)
    actions_list = actions(all_states, mines)
    policy = initalPolicy(actions_list)
    V = initalValue(all_states)

    # Value Iteration 
    iteration, data, policy = valueIteration(all_states, policy, actions_list, V, rewards_list)
    r_list = createRecords(data)
    optimal_policy = optimalPolicy(policy, start_state, end_state, iteration)
    
    # Print iterations
    print('ValueIteration took:' , iteration, ' to converge. \n')

    opt_pol = optimal_policy

    # Need landmines in the correct order:
    new_mines = reverseMines(mines)

    #Display while waiting for gif to generate
    print("Generating gif....")

    # Dispay animate from animate
    anim, fig, ax = generateAnimat(r_list, start_state, end_state, new_mines, opt_pol, start_val=-10, end_val=100, mine_val=150, just_vals=False, generate_gif=False,vmin = -10, vmax = 150, fps=1, filename="ValueIteration")
    plt.show()


