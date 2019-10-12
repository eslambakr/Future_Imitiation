# Future_Imitiation
End-To-End self driving cars using Imitation learning with future prediction using GANs.

# For Training:
1- Edit the training configruation file at Training/RGB/config.py ( the file is readable and you can adapt it easily).

2- Run the training script at Training/RGB/training.py

3- To change the data loader change: Training/RGB/single_view_generator.py

4- To change the network architecture change: Training/RGB/single_view_model.py

5- To change the network architecture change: Training/RGB/single_view_model.py

# For Data_Analysis:
Check data_analysis.ipynb

# For Testing:
1- Run driving_benchmarks/benchmarks_084.py

2- To add a new agent add it in driving_benchmarks/version084/benchmark_tools (for example I added stacking_previous_agent.py)
