# Assignment 2 - RL
    #QLearning.py

# Keagan Chasenski
# CHSKEA001
# 31st May 2021

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

# Define possible actions
# Up = 0, right = 1, down = 2, left = 3
actions = ['Up', 'Right', 'Down', 'Left']


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
    global epochs
    global learning_rate

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

        # Set the epochs value
        if "-epochs" in argv:
            enum = argv.index("-epochs")
            epochs = int(argv[enum + 1])
        # Default 1000 seems to work well
        else:
            epochs = 1000

        # Set the Learning rate value
        if "-learning" in argv:
            learning = argv.index("-learning")
            learning_rate = float(argv[learning + 1])
        # Default 0.9 seems to work well
        else:
            learning_rate = 0.9

def printArguments():
    # Simple function to print all the parameters to the terminal
    print("Width:" , width)
    print("Height:" , height)
    print("Start state:" , start_state)
    print("End state:" , end_state)
    print("Landmines:" , k)
    print("Gamma :" , g)
    print("Learning rate: ", learning_rate)
    print("Epochs: ", epochs)

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

def reverseMines(mines):
    # fuction reverses the mines x,y coords due to at the start I mixed them up
    # easier to write this function than change every previous function
    # returns reversed_mines to be displayed in the plot

    reversed_mines = []
    for m in mines:
        reversed_mines.append(m[::-1])

    return reversed_mines

def rewards():
    # Goal is to find shortest path therefor all rewards are negative except goal
    # Terminal states (ie landmines are bigger negative vlaues)

    rewards = np.full((height, width), -1)

    #set landmines -100
    for m in mines:
        rewards[m[0], m[1]] = -10

    #set end state 100
    # coords y, x
    rewards[y_end, x_end] = 100 #set the reward for the packaging area (i.e., the goal) to 100

    return rewards

def terminalState(row, column, rewards):
    #Determines if the current location is a terminal state

    #if the reward = -1, then it is not a terminal state 
    if rewards[row, column] == -10. or rewards[row, column] == 100.:
        return True
    else:
        return False

def randomStart():
    #choose a random start, not on a mine or end location
    
    row = np.random.randint(0, height-1)
    column = np.random.randint(0, width-1)

    # Loop to check that it is not terminal, choose again if it is
    while terminalState(row, column, rewards):
        row = np.random.randint(0, height -1)
        column = np.random.randint(0, width -1)
    return row, column

def nextAction(row, column):
    # Chooses the next move based of a random action

    index = np.random.randint(0,4)
    return index
    
def nextState(row, column, index):
    # Function to find next location based off the chosen action
    new_row = row
    new_column = column

    # Compared to the actions index created where Up = 0 etc. 
    if actions[index] == 'Up' and row > 0:
        new_row -= 1
    # Right = 1
    elif actions[index] == 'Right' and (column < width - 1):
        new_column += 1
    # Down = 2
    elif actions[index] == 'Down' and (row < height - 1):
        new_row += 1
    # Left = 3
    elif actions[index] == 'Left' and column > 0:
        new_column -= 1

    # Returns new state coords
    return new_row, new_column

def trainQLearning(Q, Records, rewards):
   
    # Loop for number of epochs
    for episode in range(epochs):
        # random start state
        row, column = randomStart()

        # Looping until a terminal state
        while not terminalState(row,column, rewards):    

            # Chose random action
            action_index = nextAction(row, column)

            # Do action, then transition to get state
            #store the old row and column indexes
            row_old, column_old = row, column 
            row, column = nextState(row_old, column_old, action_index)
            
            # Reward for next state
            reward = rewards[row, column]

            # Calculate temporal Differnce using old q value and the newer q value
            old_q = Q[row_old, column_old, action_index]
            TD = reward + ( g * np.max(Q[row, column])) - old_q

            #update the Q-value for the previous state and action pair
            # Updated based on Bellmans equation
            new_q = old_q + (learning_rate * TD)
            round_new_q = round(new_q, 4)
            Q[row_old, column_old, action_index] = round_new_q

            #ele = np.max(Q[row_old, column_old])
            #r.append(ele)
   

        # Convert Q to a list
        q = Q.tolist()
        #print(q)
        # Append each Q value to the Records
        Records.append(q)
    print('QLearning completed! Ran: ', epochs, " epochs \n")

    # Return Qvlaues and Records list
    return Q, Records

def optimalPolicy():
    
    #Define starting point for optimal policy
    row, column = y_start, x_start
    opt_pol = []
    opt_pol.append([column, row])

    #Move from starting untill the end state (terminal state) 
    while not terminalState(row, column, rewards):
        #get the best action 
        index = np.argmax(Q[row, column])
        #move to the next location 
        row, column = nextState(row, column, index)
        # add that location to the list
        opt_pol.append([column, row])
    
    #return the optimal policy
    return opt_pol

def createRecords(records):
    r_list = []
    # Gets each epoch of records
    for r in records:
        # Create a list for every epoch
        e = []
        # Gets every action taken until terminal state for that epoch
        for s in r:
            # List for every state
            states=[]
            # Gets the four Q values for each action at that state
            for a in s:
                # select max action for that state
                m = np.max(a)
                # append vale to list
                states.append(m)
            # Append the states to the epoch list
            e.append(states)
        # Append epoch list to r_list    
        r_list.append(e)

    #print(r_list)
    return r_list

if __name__ == "__main__" :

    # Set the arguments for running
    setArguments(sys.argv)
    start_state = (x_start, y_start)
    end_state = (x_end, y_end)
    mines = addMines()

    # Display arguments in nice format
    printArguments()
    print("Mines:" , reverseMines(mines), "\n")

    #Inital QValues
    # 3D, third dimension is a Value for each choice at that location (Max 4)
    Q = np.zeros((height, width, 4))

    #Record values 
    Records = []

    # Function Calls
    rewards = rewards()

    # QLearning Algorithm
    Qvalues, records = trainQLearning(Q, Records, rewards)

    # Creat record list to plot
    r_list = createRecords(records)

    # Optimal Policy 
    opt_pol = optimalPolicy()
    print("The optimal policy (shortest distance) from start to end: ")
    print(opt_pol, "\n")

    # Need landmines in the correct order:
    new_mines = reverseMines(mines)

    #Display while waiting for gif to generate
    print("Generating plot....")

    #Display Plot
    anim, fig, ax = generateAnimat(r_list, start_state, end_state, new_mines, opt_pol, start_val=-10, end_val=100, mine_val=150, just_vals=False, generate_gif=False,vmin = -10, vmax = 150, fps=10, filename="QLearning")
    plt.show()