

# Efficient-Code-Generation-with-E-code

![E-code模型图](https://github.com/CodeGeneration2/E-code/assets/95161813/e98eff5e-b891-4953-bb71-44aacbf39fa5)


## abstract
Context: With the waning of Moore's Law, the software industry is placing increasing importance on finding alternative solutions for continuous performance enhancement. The significance and research results of software performance optimization have been on the rise in recent years, especially with the advancement propelled by \textbf{L}arge \textbf{L}anguage \textbf{M}odel\textbf{s} (LLMs). However, traditional strategies for rectifying performance flaws have shown significant limitations at the competitive code efficiency optimization level, and research on this topic is surprisingly scarce.

Objective: This study aims to address the research gap in this domain, offering practical solutions to the various challenges encountered. Specifically, we have overcome the constraints of traditional performance error rectification strategies and developed a Language Model (LM) tailored for the competitive code efficiency optimization realm.

Method: We introduced E-code, an advanced program synthesis LM. Inspired by the recent success of expert LMs, we designed an innovative structure called the Expert Encoder Group. This structure employs multiple expert encoders to extract features tailored for different input types. We assessed the performance of E-code against other leading models on a competitive dataset and conducted in-depth ablation experiments.

Results: Upon systematic evaluation, E-code achieved a 57% improvement in code efficiency, significantly outperforming other advanced models. In the ablation experiments, we further validated the significance of the expert encoder group and other components within E-code.

Conclusion: The research findings indicate that the expert encoder group can effectively handle various inputs in efficiency optimization tasks, significantly enhancing the model's performance. In summary, this study paves new avenues for developing systems and methods to assist programmers in writing efficient code.



## ExecTimePredictor


 
To effectively predict the running time of code, we introduced the ExecTimePredictor. In the generation of
efficient code, the running time of the code serves as a crucial metric for evaluating its efficiency. A short running
time indicates efficient code, while a long running time implies inefficiency. However, a significant issue arises if the
generated code cannot compile, making it challenging to ascertain its efficiency. Given the current limitations of LMs
in ensuring the functional correctness of generated code, and the fact that code which might be fixed with minor token
adjustments still holds potential value, this paper seeks solutions to reasonably assess the efficiency of such code. To
address this issue, we introduce ExecTimePredictor, a tool capable of predicting the execution time of uncompileable
code.

We employed DeepSeek-Coder, a cutting-edge code LLM, as the foundational model for ExecTimePredictor.
DeepSeek-Coder surpasses existing code LLMs such as Code Llama and Star Coder. We selected version DeepSeekCoder-v1.5 7B, which is based on DeepSeek-LLM-7B and additionally pretrained on an extra 2 trillion tokens. To
ensure ExecTimePredictor’s effectiveness on both executable and non-executable code, we created a datasetspecifically
for fine-tuning ExecTimePredictor. We crawled and preprocessed the necessary code data, as detailed in the following
steps:

• We extracted a large volume of code data from the CodeForces website, which underwent preprocessing such
as removing code that could not generate an AST, failed I/O tests, or was excessively long.

• Given that the runtime data provided by CodeForces was not sufficiently precise, we recalculated the execution
times. By wrapping the code in functions and recording the time taken to call these functions, we significantly
enhanced the precision of our execution time measurements, converting the units from milliseconds to
microseconds.

• We divided the test set according to the submission times on the website to prevent data leakage.

Ultimately, we compiled the CodeExecTimeDB, which contains a total of 147,677 entries, 16,662 of which are
designated as the test set. Depending on different needs, we created four distinct dataset variants, as follows:

• CodeExecTimeDB-Ori: The original version, without any modifications.

• CodeExecTimeDB-Uni: Builds on CodeExecTimeDB-Ori by standardizing variable and function names (e.g.,
renaming variables to var1). Since differences in variable and function names do not affect execution time, this
helpsto reduce noise interference. Additionally, asthe GEC dataset isinsensitive to variable and function names,
the fine-tuned model produces code with uniform naming conventions. Therefore, ExecTimePredictor is trained
to handle such code.

• CodeExecTimeDB-Loop&Rec: Based on CodeExecTimeDB-Uni, retains only loop and recursion statements.
These elements often account for a significant portion of the code’s execution time and remain relevant even if
the overall code is non-executable. This variant aims to teach the model to infer the overall code execution time
solely from loops and recursion.

• CodeExecTimeDB-RandDel: Builds on CodeExecTimeDB-Uni by randomly deleting 20% of code tokens. This
random deletion mimics non-executable code more closely, further ensuring ExecTimePredictor’s effectiveness
on such code.



### CodeExecTimeDB
CodeExecTimeDB are [here](https://drive.google.com/file/d/1-JoO4ziUFRmntMkPzj_nxwTzuPxRecyT/view?usp=sharing).



## Fine-tuning ExecTimePredictor

To develop the ExecTimePredictor, we fine-tuned the DeepSeek-Coder-v1.5 7B using the AdamW optimizer
with a batch size of 32. The model was fine-tuned over four datasets, each for two epochs. After this process, we
obtained the refined ExecTimePredictor. The fine-tuned ExecTimePredictor demonstrated accuracies of 8.50 µs, 8.11
µs, 8.35 µs, and 8.21 µs on the CodeExecTimeDB-Ori, CodeExecTimeDB-Uni, CodeExecTimeDB-Loop&Rec, and
CodeExecTimeDB-RandDel test datasets, respectively.

During the fine-tuning and prediction phases with DeepSeek-Coder-v1.5 7B, we designed the following prompt,
inspired by the chat template provided by DeepSeek, to guide the model in predicting the execution time of code:

“### Code Solution:\n{Code}\n\n### Predict the running time of the provided code solution:”


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




