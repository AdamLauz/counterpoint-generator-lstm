# counterpoint-generator-lstm
Generate counter point melody given cantus firmus using LSTM model. 
This project is based on [[1]](#1) by Valerio Velardo.

## Model
LSTM (Long Short-Term Memory) [[2]](#2) model is trained on sequences of cantus firmus and counter point melodies.

## Data
MusicXML Dataset for training the LSTM model is from http://www.mscorelib.com.
Sample of about 100 Bach's pieces are included in this project.

## Input-Output Example

**Input**: Sub-section of "Autumn Leaves" by Joseph Kosma:
![image](https://github.com/AdamLauz/counterpoint-generator-lstm/assets/2620814/c5ab5d14-b526-41c2-b34b-91ccbb80800a)

**Output**: The input melody with counter point in Bach's style
![image](https://github.com/AdamLauz/counterpoint-generator-lstm/assets/2620814/f7b2a50a-9e6b-435c-af83-9206130c74b9)


## References
<a id="1">[1]</a>
Valerio Velardo.
https://github.com/musikalkemist/generating-melodies-with-rnn-lstm

<a id="2">[2]</a> 
Hochreiter, S. and Schmidhuber, J., 1997. 
Long short-term memory.
Neural computation, 9(8), pp.1735-1780.
