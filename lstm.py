import numpy as np

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def sigmoid_derivative(values):
    return values*(1 - values)

def tanh_derivative(values):
    return 1. - values ** 2

def random_array(a, b, *args):
    return np.random.default_rng().random(size=args) * (b - a) + a

class LstmParameters:
    def __init__(self, memory_cell_count, input_dimensions):
        self.memory_cell_count = memory_cell_count
        self.input_dimensions = input_dimensions
        concatenated_length = input_dimensions + memory_cell_count

        # weight matrices
        self.w_generate = random_array(-0.1, 0.1, memory_cell_count, concatenated_length)
        self.w_input = random_array(-0.1, 0.1, memory_cell_count, concatenated_length)
        self.w_forget = random_array(-0.1, 0.1, memory_cell_count, concatenated_length)
        self.w_output = random_array(-0.1, 0.1, memory_cell_count, concatenated_length)

        # bias terms
        self.b_generate = random_array(-0.1, 0.1, memory_cell_count)
        self.b_input = random_array(-0.1, 0.1, memory_cell_count)
        self.b_forget = random_array(-0.1, 0.1, memory_cell_count)          
        self.b_output = random_array(-0.1, 0.1, memory_cell_count)

        self.W_y = np.random.uniform(-0.1, 0.1, (1, memory_cell_count))
        self.b_Y = np.random.uniform(-0.1, 0.1, (1,))

        # diffs (derivative of loss function with respect to all parameters)
        self.w_generate_diff = np.zeros((memory_cell_count, concatenated_length))
        self.w_input_diff = np.zeros((memory_cell_count, concatenated_length))
        self.w_forget_diff = np.zeros((memory_cell_count, concatenated_length))
        self.w_output_diff = np.zeros((memory_cell_count, concatenated_length))
        self.b_generate_diff = np.zeros(memory_cell_count)
        self.b_input_diff = np.zeros(memory_cell_count)
        self.b_forget_diff = np.zeros(memory_cell_count)
        self.b_output_diff = np.zeros(memory_cell_count)

    def apply_derivatives(self, learning_rate = 1):
        self.w_generate -= learning_rate * self.w_generate_diff
        self.w_input -= learning_rate * self.w_input_diff
        self.w_forget -= learning_rate * self.w_forget_diff
        self.w_output -= learning_rate * self.w_output_diff
        self.b_generate -= learning_rate * self.b_generate_diff
        self.b_input -= learning_rate * self.b_input_diff
        self.b_forget -= learning_rate * self.b_forget_diff
        self.b_output -= learning_rate * self.b_output_diff
        # reset diffs to zero
        self.w_generate_diff = np.zeros_like(self.w_generate)
        self.w_input_diff = np.zeros_like(self.w_input)
        self.w_forget_diff = np.zeros_like(self.w_forget)
        self.w_output_diff = np.zeros_like(self.w_output)
        self.b_generate_diff = np.zeros_like(self.b_generate)
        self.b_input_diff = np.zeros_like(self.b_input)
        self.b_forget_diff = np.zeros_like(self.b_forget)
        self.b_output_diff = np.zeros_like(self.b_output)

class LstmState:
    def __init__(self, memory_cell_count, input_dimensions):
        self.g = np.zeros(memory_cell_count)
        self.i = np.zeros(memory_cell_count)
        self.f = np.zeros(memory_cell_count)
        self.o = np.zeros(memory_cell_count)
        self.s = np.zeros(memory_cell_count) # cell state
        self.h = np.zeros(memory_cell_count) # hidden state
        self.bottom_diff_h = np.zeros_like(self.h) # Gradient of loss with respect to hidden state
        self.bottom_diff_s = np.zeros_like(self.s) # Gradient of loss with respect to cell state

class LstmNode:
    def __init__(self, lstm_parameters, lstm_state):
        # store reference to parameters and to activations
        self.state = lstm_state
        self.param = lstm_parameters
        # store concatenated input (current input + previous hidden state)
        self.xc = None

    # forward pass (computes the LSTM’s hidden state and cell state at time)
    def bottom_data_is(self, current_input_vector, s_prev = None, h_prev = None):
        # if this is the first lstm node in the network
        if s_prev is None: s_prev = np.zeros_like(self.state.s)
        if h_prev is None: h_prev = np.zeros_like(self.state.h)
        # save data for use in backprop
        self.s_prev = s_prev
        self.h_prev = h_prev

        # concatenate current input vector "x(t)" and h(t-1)
        xc = np.hstack((current_input_vector, h_prev))
        self.state.g = np.tanh(np.dot(self.param.w_generate, xc) + self.param.b_generate)
        self.state.i = sigmoid(np.dot(self.param.w_input, xc) + self.param.b_input)
        self.state.f = sigmoid(np.dot(self.param.w_forget, xc) + self.param.b_forget)
        self.state.o = sigmoid(np.dot(self.param.w_output, xc) + self.param.b_output)
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        self.state.h = self.state.s * self.state.o

        self.xc = xc

    # backward pass (computes gradients of loss with respect to parameters for backpropagation)
    def top_diff_is(self, top_diff_h, top_diff_s):
        gradient_s = self.state.o * top_diff_h + top_diff_s
        gradient_o = self.state.s * top_diff_h
        gradient_i = self.state.g * gradient_s
        gradient_g = self.state.i * gradient_s
        gradient_f = self.s_prev * gradient_s

        # diffs with respect to vector inside sigma / tanh function
        gradient_i_input = sigmoid_derivative(self.state.i) * gradient_i
        gradient_f_input = sigmoid_derivative(self.state.f) * gradient_f
        gradient_o_input = sigmoid_derivative(self.state.o) * gradient_o
        gradient_g_input = tanh_derivative(self.state.g) * gradient_g

        # diffs with respect to inputs
        self.param.w_input_diff += np.outer(gradient_i_input, self.xc)
        self.param.w_forget_diff += np.outer(gradient_f_input, self.xc)
        self.param.w_output_diff += np.outer(gradient_o_input, self.xc)
        self.param.w_generate_diff += np.outer(gradient_g_input, self.xc)
        self.param.b_input_diff += gradient_i_input
        self.param.b_forget_diff += gradient_f_input
        self.param.b_output_diff += gradient_o_input
        self.param.b_generate_diff += gradient_g_input

        # compute bottom diff
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.w_input.T, gradient_i_input)
        dxc += np.dot(self.param.w_forget.T, gradient_f_input)
        dxc += np.dot(self.param.w_output.T, gradient_o_input)
        dxc += np.dot(self.param.w_generate.T, gradient_g_input)

        # save bottom diffs
        self.state.bottom_diff_s = gradient_s * self.state.f
        self.state.bottom_diff_h = dxc[self.param.input_dimensions:]


class LstmNetwork():
    def __init__(self, lstm_parameters, memory_cell_count=100):
        self.lstm_parameters = lstm_parameters or LstmParameters(memory_cell_count)
        self.lstm_node_list = []
        # input sequence
        self.x_list = []

        # Initialize weights only if not provided
        if lstm_parameters is None:
            self.W_y = np.random.uniform(-0.1, 0.1, (1, memory_cell_count))
            self.b_Y = np.random.uniform(-0.1, 0.1, (1,))
        else:
            self.W_y = lstm_parameters.W_y  # Load from parameters if provided
            self.b_Y = lstm_parameters.b_Y  

        self.W_y_diff = np.zeros_like(self.W_y)
        self.b_Y_diff = np.zeros_like(self.b_Y)

    def reset_state(self):
        """Reset LSTM state before a new prediction."""
        self.lstm_node_list.clear()
        self.x_list.clear()

    def get_output(self):
        """Compute y = W_y * h + b_y"""
        h_last = self.lstm_node_list[-1].state.h
        return np.dot(self.W_y, h_last) + self.b_Y
    
    def update_output_layer(self, learning_rate=0.01):
        """Apply gradients to W_y and b_Y"""
        self.W_y -= learning_rate * self.W_y_diff
        self.b_Y -= learning_rate * self.b_Y_diff

        # Reset gradients
        self.W_y_diff = np.zeros_like(self.W_y)
        self.b_Y_diff = np.zeros_like(self.b_Y)

    def y_list_is(self, y_list, loss_layer):
        """
        Updates diffs by setting target sequence 
        with corresponding loss layer. 
        Will *NOT* update parameters.  To update parameters,
        call self.lstm_param.apply_diff()
        """
        assert len(y_list) == len(self.x_list)
        idx = len(self.x_list) - 1

        loss = loss_layer.loss(self.get_output(), y_list[idx])
        diff_y = loss_layer.bottom_diff(self.get_output(), y_list[idx]) # Gradient wrt y

        # Backprop through output layer
        self.W_y_diff += np.outer(diff_y, self.lstm_node_list[-1].state.h)
        self.b_Y_diff += diff_y

        diff_h = np.dot(self.W_y.T, diff_y) # Gradient wrt h

        # Backprop through lstm layers
        diff_s = np.zeros(self.lstm_parameters.memory_cell_count)
        self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)

        ### ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
        ### we also propagate error along constant error carousel using diff_s
        while idx > 0:
            loss += loss_layer.loss(self.get_output(), y_list[idx])
            diff_y = loss_layer.bottom_diff(self.get_output(), y_list[idx])
            self.W_y_diff += np.outer(diff_y, self.lstm_node_list[idx].state.h)
            self.b_Y_diff += diff_y
            diff_h = np.dot(self.W_y.T, diff_y) + self.lstm_node_list[idx].state.bottom_diff_h
            diff_s = self.lstm_node_list[idx].state.bottom_diff_s
            self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
            idx -= 1 

        return np.mean(loss)
    
    def x_list_clear(self):
        self.x_list = []

    def x_list_add(self, value):
        self.x_list.append(value)
        if len(self.x_list) > len(self.lstm_node_list):
            # need to add new lstm node, create new state mem
            lstm_state = LstmState(self.lstm_parameters.memory_cell_count, self.lstm_parameters.input_dimensions)
            self.lstm_node_list.append(LstmNode(self.lstm_parameters, lstm_state))

        # get index of most recent input
        idx = len(self.x_list) - 1
        if idx == 0:
            # no recurrent inputs yet 
            self.lstm_node_list[idx].bottom_data_is(value)
        else:
            s_prev = self.lstm_node_list[idx - 1].state.s
            h_prev = self.lstm_node_list[idx - 1].state.h
            self.lstm_node_list[idx].bottom_data_is(value, s_prev, h_prev)