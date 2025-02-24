#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np;
import pickle;


# In[36]:


BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_BLOCKS = 3
FIRST_PLAYER_SYMBOL = 1;
SECOND_PLAYER_SYMBOL = -1;
HUMAN_PLAYER_SYMBOL = 0;


# In[74]:


class HumanPlayer:
    def __init__(self):
        self.name = 'human';
    
    def chooseAction(self, positions, _borad, _symbol, _isRandom):
        while True:
            block = int(input("Input your action block (from 1 to 3):"))
            row = int(input("Input your action row (from 1 to 3):"))
            column = int(input("Input your action col (from 1 to 3):"))
            action = (block - 1, row - 1, column - 1);
            if action in positions:
                return action
    
    # append a hash state
    def addState(self, state):
        pass
    
    def feedReward(self, reward):
        pass
            
    def reset(self):
        pass


# In[75]:


class ComputerPlayer:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.3
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}  # state -> value

    def rotate_board(self, board, axis):
        return np.rot90(board, k=1, axes=axis)
    
    # Define a function to reflect the board across an axis
    def reflect_board(self, board, axis):
        return np.flip(board, axis=axis)
    
    def generate_symmetric_states(self, board):
        symmetric_states = []
        axes = [(1, 2), (0, 2), (0, 1)]  # Rotate around different pairs of axes
        
        for axis in axes:
            for _ in range(4):  # Four 90-degree rotations
                board = self.rotate_board(board, axis)
                symmetric_states.append(board)
                # Reflect along each axis after rotating
                symmetric_states.append(self.reflect_board(board, axis=0))
                symmetric_states.append(self.reflect_board(board, axis=1))
                symmetric_states.append(self.reflect_board(board, axis=2))
        
        return symmetric_states
    
    # Choose the canonical state
    def get_canonical_state(self, board):
        symmetric_states = self.generate_symmetric_states(board)
        # Convert states to lists and find the lexicographically smallest one
        canonical_state = min(symmetric_states, key=lambda x: x.flatten().tolist())
        return canonical_state;
    
    # get unique hash of current board state
    def getHash(self, board):
        return str(self.get_canonical_state(board));
    
    def chooseAction(self, positions, current_board, symbol, isRandom = True, trainingProgress = 1):
        if isRandom and np.random.uniform(0, 1) <= (1 - (1 - self.exp_rate) * trainingProgress):
            # take random action
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
    
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardHash = self.getHash(next_board);
                
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
 
                if value >= value_max:
                    value_max = value
                    action = p

        return action
    
    # append a hash state
    def addState(self, state):
        self.states.append(state)
    
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0;
            self.states_value[st] += self.lr*(self.decay_gamma*reward - self.states_value[st]);
            
            reward = self.states_value[st];
            
    def reset(self):
        self.states = [];
        
    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file,'rb')
        self.states_value = pickle.load(fr);
        fr.close()


# In[76]:


class Game:
    def __init__(self):
        self.board = np.zeros((BOARD_BLOCKS, BOARD_ROWS, BOARD_COLS))
        self.p1 = ComputerPlayer("p1")
        self.p2 = ComputerPlayer("p2");
        self.p3 = HumanPlayer();
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first
        self.playerSymbol = FIRST_PLAYER_SYMBOL

    def rotate_board(self, board, axis):
        return np.rot90(board, k=1, axes=axis)
    
    # Define a function to reflect the board across an axis
    def reflect_board(self, board, axis):
        return np.flip(board, axis=axis)
    
    # Generate all symmetric states (rotations and reflections)
    def generate_symmetric_states(self, board):
        symmetric_states = []
        axes = [(1, 2), (0, 2), (0, 1)]  # Rotate around different pairs of axes
        
        for axis in axes:
            for _ in range(4):  # Four 90-degree rotations
                board = self.rotate_board(board, axis)
                symmetric_states.append(board)
                # Reflect along each axis after rotating
                symmetric_states.append(self.reflect_board(board, axis=0))
                symmetric_states.append(self.reflect_board(board, axis=1))
                symmetric_states.append(self.reflect_board(board, axis=2))
        
        return symmetric_states
    
    # Choose the canonical state
    def get_canonical_state(self, board):
        symmetric_states = self.generate_symmetric_states(board)
        # Convert states to lists and find the lexicographically smallest one
        canonical_state = min(symmetric_states, key=lambda x: x.flatten().tolist())
        return canonical_state;
    
    # get unique hash of current board state
    def getHash(self):
        self.boardHash = str(self.get_canonical_state(self.board));
        
        return self.boardHash
    
    def winner(self):
        def check_line(line):
            if sum(line) == FIRST_PLAYER_SYMBOL * 3:
                return FIRST_PLAYER_SYMBOL  # Player 1 wins
            elif sum(line) == SECOND_PLAYER_SYMBOL * 3:
                return SECOND_PLAYER_SYMBOL  # Player 2 wins
            return None  # No winner yet
        # Check all 2D layers (xy-plane)
        board = self.board;
        for layer in range(BOARD_ROWS):
            # Check rows and columns in each layer
            for i in range(BOARD_ROWS):
                if check_line(board[layer, i, :]):  # Check row in layer
                    return check_line(board[layer, i, :])
                if check_line(board[layer, :, i]):  # Check column in layer
                    return check_line(board[layer, :, i])
            # Check diagonals in each layer
            if check_line(np.diag(board[layer])):  # Main diagonal
                return check_line(np.diag(board[layer]))
            if check_line(np.diag(np.fliplr(board[layer]))):  # Anti-diagonal
                return check_line(np.diag(np.fliplr(board[layer])))
    
        # Check across layers (z-axis)
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if check_line(board[:, i, j]):  # Check columns across layers
                    return check_line(board[:, i, j])
    
        # Check 3D diagonals (spanning layers)
        if check_line(np.array([board[i, i, i] for i in range(BOARD_ROWS)])):  # Main 3D diagonal
            return check_line(np.array([board[i, i, i] for i in range(BOARD_ROWS)]))
        if check_line(np.array([board[i, i, BOARD_ROWS - 1 - i] for i in range(BOARD_ROWS)])):  # Anti 3D diagonal
            return check_line(np.array([board[i, i, BOARD_ROWS - 1 - i] for i in range(BOARD_ROWS)]))
        if check_line(np.array([board[i, BOARD_ROWS - 1 - i, i] for i in range(BOARD_ROWS)])):  # Another 3D diagonal
            return check_line(np.array([board[i, BOARD_ROWS - 1 - i, i] for i in range(BOARD_ROWS)]))
        if check_line(np.array([board[BOARD_ROWS - 1 - i, i, i] for i in range(BOARD_ROWS)])):  # Another 3D diagonal
            return check_line(np.array([board[BOARD_ROWS - 1 - i, i, i] for i in range(BOARD_ROWS)]))
        
        # tie
        # no available positions
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        # not end
        self.isEnd = False
        return None
    
    def availablePositions(self):
        positions = [];
        for i in range(BOARD_BLOCKS):
            for j in range(BOARD_ROWS):
                for k in range(BOARD_COLS):
                    if self.board[i, j, k] == 0:
                        positions.append((i, j, k))  # need to be tuple
        return positions
    
    def updateState(self, position):
        self.board[position] = self.playerSymbol
        # switch to another player
        self.playerSymbol = SECOND_PLAYER_SYMBOL if self.playerSymbol == FIRST_PLAYER_SYMBOL else FIRST_PLAYER_SYMBOL
    
    # only when game ends
    def giveEndReward(self):
        result = self.winner()
        # backpropagate reward
        if result == FIRST_PLAYER_SYMBOL:
            self.p1.feedReward(1)
            self.p2.feedReward(-0.1)
        elif result == SECOND_PLAYER_SYMBOL:
            self.p1.feedReward(-0.1)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.5)
            self.p2.feedReward(1)

    # only when play with human
    def giveTestEndReward(self, loser):
        loser.feedReward(-0.1);

    def giveProcessReward(self, step, currentSymbol):
        if currentSymbol == HUMAN_PLAYER_SYMBOL:
            return;
        
        if currentSymbol == FIRST_PLAYER_SYMBOL and len(self.availablePositions()) == BOARD_BLOCKS * BOARD_ROWS * BOARD_COLS - 1 and step == (1, 1, 1):
            self.p1.feedReward(0.8);
            return;

        # Define a function to check if a move blocks the opponent
        opponent = SECOND_PLAYER_SYMBOL if currentSymbol == FIRST_PLAYER_SYMBOL else FIRST_PLAYER_SYMBOL  # Opponent's mark is the opposite of current player's mark
        block, row, col = step;

        # Define a helper function to check if the opponent was about to win
        def check_line(line):
            output = line.tolist().count(opponent) == 2 and line.tolist().count(currentSymbol) == 1
            return output;
    
        # Check if the current move blocked any of the opponent's possible lines
        blocked = False
        
        # Check the row in the current block
        if check_line(self.board[block, row, :]):
            blocked = True
        
        # Check the column in the current block
        if check_line(self.board[block, :, col]):
            blocked = True
        
        # Check the depth (blocks) along the same row and column
        if check_line(self.board[:, row, col]):
            blocked = True
        
        # Check diagonals within the current block
        if row == col and check_line(np.diagonal(self.board[block, :, :])):
            blocked = True
        if row + col == 2 and check_line(np.diagonal(np.fliplr(self.board[block, :, :]))):
            blocked = True
        
        # Check 3D diagonals
        if block == row == col and check_line(np.array([self.board[i, i, i] for i in range(3)])):
            blocked = True
        if block == row == 2 - col and check_line(np.array([self.board[i, i, 2 - i] for i in range(3)])):
            blocked = True
        
        if blocked:
            if currentSymbol == FIRST_PLAYER_SYMBOL:
                self.p1.feedReward(0.8);
            else:
                self.p2.feedReward(0.8);
            
            
    
    # board reset
    def reset(self):
        self.board = np.zeros((BOARD_BLOCKS, BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = FIRST_PLAYER_SYMBOL
        self.p1.reset();
        self.p2.reset();
    
    def train(self, rounds=100):
        print("training...");
    
        for i in range(rounds):
            
            trainingProgress = i / rounds;
            current_player, next_player = self.p1, self.p2;
            current_symbol, next_symbol = (FIRST_PLAYER_SYMBOL, SECOND_PLAYER_SYMBOL)
            
            if i%1000 == 0:
                print("Rounds {}".format(i));
                
            self.reset();
            
            while not self.isEnd:
                
                positions = self.availablePositions()
                action = current_player.chooseAction(positions, self.board, current_symbol, True, trainingProgress)

                # Update board state
                
                self.updateState(action)
                board_hash = self.getHash()
                current_player.addState(board_hash)
                self.giveProcessReward(action, current_symbol)

                # Check for a winner
                win = self.winner()
                if win is not None:
                    self.giveEndReward()
                    self.reset()  # Reset for a new game
                    self.p1.reset();
                    self.p2.reset();
                    break;

                # Switch players
                current_player, next_player = next_player, current_player
                current_symbol, next_symbol = next_symbol, current_symbol  
        self.p1.savePolicy();
        self.p2.savePolicy();
        print("Training end");

    # play with human
    def play(self, isHumanFirst = False):
        if isHumanFirst:
            current_player, next_player = self.p3, self.p2;
            current_symbol, next_symbol = (HUMAN_PLAYER_SYMBOL, SECOND_PLAYER_SYMBOL);
            self.p2.loadPolicy('policy_p2');
            print('Human: X    |      Computer: O');
        else:
            current_player, next_player = self.p1, self.p3;
            current_symbol, next_symbol = (FIRST_PLAYER_SYMBOL, HUMAN_PLAYER_SYMBOL);
            self.p1.loadPolicy('policy_p1');
            print('Human: O    |      Computer: X');
        

        while not self.isEnd:
            for player, symbol in [(current_player, current_symbol), (next_player, next_symbol)]:
                positions = self.availablePositions()
                action = player.chooseAction(positions, self.board, symbol, False)

                # Update board state
                
                self.updateState(action)
                board_hash = self.getHash();
                self.showBoard(player, action);
                current_player.addState(board_hash)
                self.giveProcessReward(action, symbol)

                # Check for a winner
                win = self.winner()
                if win is not None:
                    if win != 0:
                        if current_player.name == 'human':
                            self.giveTestEndReward(next_player);
                        print(f"{current_player.name} wins!")
                    else:
                        print("It's a tie!");

                    if isHumanFirst:
                        self.p2.savePolicy();
                    else:
                        self.p1.savePolicy();
                    
                    self.reset()  # Reset for a new game
                    return  # Exit once game ends

                # Switch players
                current_player, next_player = next_player, current_player
                current_symbol, next_symbol = next_symbol, current_symbol  
     
                
    def showBoard(self, player, action):
        block, row, col = action;
        # p1: x  p2: o
        print('-----------------------');
        print(f"It's {player.name}'s trun to play")
        for r in range(3):
            for b in range(3):
                for c in range(3):
                    if self.board[b][r, c] == FIRST_PLAYER_SYMBOL:
                        token = 'x'
                    if self.board[b][r, c] == SECOND_PLAYER_SYMBOL:
                        token = 'o'
                    if self.board[b][r, c] == 0:
                        token = '-'

                    if block == b and row == r and col == c:
                        token = token.upper();
                    print(token, end=' ')
                print('   ', end='')
            print();


# In[77]:


game = Game();


game.train(100000);

# In[86]:


game.reset();
game.play(True);


# In[ ]:




