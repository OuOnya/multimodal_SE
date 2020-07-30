# multimodal_SE


This model is written in pytorch 1.4.0. You first need to install some packages:
```
pip install -r requirements.txt
```
or
```
conda install --yes --file requirements.txt
```

## Training
To train the model, you need to specify the **Dataset** and **raw data** folders. The data set stores clean and noisy speech, and the raw data stores electrical signals. These sentences are recorded based on the Taiwan Mandarin hearing in noise test (Taiwan MHINT). Then set the parameters you want in [``config.py``](config.py).
For the model structure, you can set the parameters of the function [S_CNN()](main.py#L131) called in main.py.
Then run it:
```
python main.py
```

## Testing & Visualization
To test the model, [``test.py``](test.py) can show the performance and the spectrogram of some specific models in specific test sample.
```
python test.py
```

[``analyze.py``](analyze.py) can calculate the average of PESQ, STOI and ESTOI. It can also show the performance of the selected model a bar chart.
```
python analyze.py
```
