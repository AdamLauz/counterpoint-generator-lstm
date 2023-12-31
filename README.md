# counterpoint-generator-lstm
Generate counter point melody given cantus firmus using LSTM model. 
This project is based on [[1]](#1) by Valerio Velardo.

## Model
LSTM (Long Short-Term Memory) [[2]](#2) model is trained on sequences of cantus firmus and counter point melodies.

## Data
MusicXML Dataset for training the LSTM model is from http://www.mscorelib.com.
Sample of about 100 Bach's pieces are included in this project.

## Input-Output Example

**Input**: Sub-section of "Autumn Leaves" by Joseph Kosma ("cantus.mxl"):
![image](https://github.com/AdamLauz/counterpoint-generator-lstm/assets/2620814/c5ab5d14-b526-41c2-b34b-91ccbb80800a)

**Output**: The input melody with counter point in Bach's style ("melody.mid")
![image](https://github.com/AdamLauz/counterpoint-generator-lstm/assets/2620814/9944594d-0d21-44ea-9ef3-224479065bbf)

## Video Demo
[<img src="https://img.youtube.com/vi/A5JWlJrUqtk/hqdefault.jpg" width="800" height="600"
/>](https://www.youtube.com/embed/A5JWlJrUqtk)


## References
<a id="1">[1]</a>
Valerio Velardo.
https://github.com/musikalkemist/generating-melodies-with-rnn-lstm

<a id="2">[2]</a> 
Hochreiter, S. and Schmidhuber, J., 1997. 
Long short-term memory.
Neural computation, 9(8), pp.1735-1780.
