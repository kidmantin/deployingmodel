TensorFlow model which recognises audio input of 8 commands, deployed using bentoml, streamlit (UI) and Heroku


link to deployed model: https://deployingmodel-gljy64kms8v.streamlit.app/


(NOTE: Dyno on Heroku may be turned off, in that case button 'predict' won't work)


In this case you can find an example of the work below:


Front page:


![frontpage](https://github.com/kidmantin/deployingmodel/blob/main/images/startUI.jpg?raw=true)


As input you can use your voice, for that press the microphone button


After that you can either playback your input, or choose between 2 options  (as example I've said UP):
- get waveform and spectrogram plots
![examlpePLOT_UP](https://github.com/kidmantin/deployingmodel/blob/main/images/examplePLOT_UP.jpg?raw=true)
- get prediction in the form of a probability bar chart
![exampleUP](https://github.com/kidmantin/deployingmodel/blob/main/images/exampleUP.jpg?raw=true)


More examples of prediction output:
![exampleRIGHT](https://github.com/kidmantin/deployingmodel/blob/main/images/exampleRIGHT.jpg?raw=true)
![exampleSTOP](https://github.com/kidmantin/deployingmodel/blob/main/images/exampleSTOP.jpg?raw=true)
![exampleYES](https://github.com/kidmantin/deployingmodel/blob/main/images/exampleYES.jpg?raw=true)
