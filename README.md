### CNN-LSTM-Attention model for water quality prediction
- The **data** folder contains the data after wavelet denoising. 
- The **models** folder contains all of our models.
- CLA20.py is our proposed model, and the default value in the code is to predict the pH value. 
- If you need to predict NH3-N, you need to change the parameters: 
  - you need to change the parameters of the model
  - then change 0 to 3 in line 73 of data_processing.py.