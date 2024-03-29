

# Efficient-Code-Generation-with-E-code

![E-code模型图](https://github.com/CodeGeneration2/E-code/assets/95161813/e98eff5e-b891-4953-bb71-44aacbf39fa5)





## How to Use

### Implementation Train the model -> predict the generated code -> perform IO test on the generated code.
#### To use the E_code source code extremely fast: 

1. Extract the GEC dataset to the E_code folder and change the file name to GEC. 
2. Run the train.py file. 

#### Fast-running classification experiments: 

Set Command_line_parameters.task = 0 to train the E-code model.

Set Command_line_parameters.task = 0 and set Command_line_parameters.RELU = 1 to train a comparison experiment using the RELU activation function.

Set Command_line_parameters.task = 0 and set Command_line_parameters. heads = 8 to train a comparison experiment using 8 heads.

Set Command_line_parameters.task = 1 to train the No-expert-E-code model.


#### Extremely fast use of Time_Predictor source code: 
1. Extract the GEC dataset to the E_code folder and change the file name to GEC. 
2. Run the train.py file to train the model.

3. Put the code to be predicted into Code_to_be_predicted a
4. Run Prediction_generation_code to automatically predict the code runtime.


## Model parameters
All model parameters are [here](https://drive.google.com/drive/folders/18tg9mTBZ3E6bmpnoelMbYqMo_o3B76bX?usp=sharing).

## CodeExecTimeDB
CodeExecTimeDB are [here](https://drive.google.com/file/d/1tR3R9Mf9thXBUszMo34Pmdli0K4savjp/view?usp=sharing).
