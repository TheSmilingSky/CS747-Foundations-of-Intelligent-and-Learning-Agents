#! /bin/sh
echo "Running Baseline Simulations"
python3 ../codes/gridworld.py
echo "Plot for Baseline Simulations Saved"
echo "-----------------------------------"
echo "Running Simulations for environment with King Moves allowed"
python3 ../codes/gridworld.py -env gridking
echo "Plot for King Moves Simulations Saved"
echo "-----------------------------------"
echo "Running Simulations for stochastic environment with King Moves allowed"
python3 ../codes/gridworld.py -env gridking -sto True
echo "Plot for Stochastic King Moves Simulations Saved"
echo "-----------------------------------"
echo "Running Comparision between different algorithms for baseline environment"
python3 ../codes/gridworld.py -algo 'Sarsa,Qlearning,ExpectedSarsa'
echo "Plot for Comparision of algorithms saved"
echo "-----------------------------------"
echo "Simulation Complete. All plots have been saved"
