# counterpoint-generator-lstm
Generate counter point melody given cantus firmus using LSTM model. 
This project is based on [[1]](#1) by Valerio Velardo.

## Model
LSTM (Long Short-Term Memory) [[2]](#2) model is trained on sequences of cantus firmus and counter point melodies.

## Data
Dataset for training the LSTM model is from http://www.mscorelib.com.
Sample of about 100 Bach's pieces are included in this project.

## References
<a id="1">[1]</a>
Valerio Velardo.
https://github.com/musikalkemist/generating-melodies-with-rnn-lstm

<a id="2">[2]</a> 
Hochreiter, S. and Schmidhuber, J., 1997. 
Long short-term memory.
Neural computation, 9(8), pp.1735-1780.
