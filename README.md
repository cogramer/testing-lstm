# testing-lstm
LSTM implementation based on nicodjimenez's example.
This was only made for educational purposes. To see the original code, please check out nicodjimenez's repository on lstm.

-
# lstm

A basic lstm network can be written from scratch in a few hundred lines of python, yet most of us have a hard time figuring out how lstm's actually work. The original Neural Computation paper is too technical for non experts. Most blogs online on the topic seem to be written by people who have never implemented lstm's for people who will not implement them either. Other blogs are written by experts (like this blog post) and lack simplified illustrative source code that actually does something. The Apollo library built on top of caffe is terrific and features a fast lstm implementation. However, the downside of efficient implementations is that the source code is hard to follow.

This repo features a minimal lstm implementation for people that are curious about lstms to the point of wanting to know how lstm's might be implemented. The code here follows notational conventions set forth in this well written tutorial introduction. This article should be read before trying to understand this code (at least the part about lstm's). By running python test.py you will have a minimal example of an lstm network learning to predict an output sequence of numbers in [-1,1] by using a Euclidean loss on the first element of each node's hidden layer.

Play with code, add functionality, and try it on different datasets. Pull requests welcome.

Please read my blog article if you want details on the backprop part of the code.

This sample code has been ported to the D programming language by Mathias Baumann: https://github.com/Marenz/lstm as well as Julia by @hyperdo https://github.com/hyperdo/julia-lstm, Alfiuman in C++ (with CUDA) https://github.com/Alfiuman/WhydahGally-LSTM-MLP and Ascari in JavaScript (for nodejs) https://github.com/carlosascari/lstm.
