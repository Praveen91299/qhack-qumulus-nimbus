# Feeding many trolls  
## Team Qumulus Nimbus  
praveen91299@gmail.com  
My team submission for QHACK 2021, hosted by Xanadu.ai  

## Last updated log:  
27th Feb 4:30AM IST

## Content:  
Jupyter notebook contains information of methods used and their explanations  
qumulus.py provides three music learning and generation functions for different geometries. Refer to the jupyter notebook for more details.  

## Project abstract:  
We provide a pennylane implementation of single qubit universal quantum classifier similar to that presented in [1] and [2]. We then provide an efficient method to parallely process classical data using a qram setup for the universal single qubit classifier.  
We then attempt to address quantum classifiers by data reuploading for **Quantum Data** for experiments when we have copies of the quantum state and show it's performance, which we believe has not been done before. We show that it does not perform any better than a single copy.  

We use the universal quantum classifier method and measurement strategies described in [1] to demonstrate a method of quantum music learning and generation by recasting the classifier into a markov chain like setup. We also use the qram structure we developed to attempt to combine and generate music of a similar scale and style. 

## Inspiration:  
The ideas in this project were directly inspired by the talk https://www.twitch.tv/videos/921036560 and ideas expressed in [1]. Our initial goal was to address the open problem of quantum data as stated by the speaker and then divulged to the current state. Also I guess the idea to qram the input stems from my experience with the Qiskit 2020 challenge.  

## Future work:  
Work into the music generation mappings, and different geometries on sphere.

## References:  
[1] - https://quantum-journal.org/papers/q-2020-02-06-226/  
[2] - https://github.com/AlbaCL/qhack21f  
[3] - https://journals.aps.org/pra/abstract/10.1103/PhysRevA.102.032420  
