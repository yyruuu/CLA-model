### CNN-LSTM-Attention model for water quality prediction
- The **data** folder contains the data after wavelet denoising. 
- The **models** folder contains all of our models.
- CLA20.py is our proposed model, and the default value in the code is to predict the pH value. 
- If you need to predict NH3-N, you need to unncomment `values[['0', '3']] = values[['3', '0']]`.