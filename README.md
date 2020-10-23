# STSVB

MATLAB implementation of Spatio-Temporal Sparse Variational Bayes Framework. For more details, please refer to the paper: 

Shruti Sharma, Santanu Chaudhury and Jayadeva, "Block Sparse Variational Bayes Regression Using Matrix Variate Distributions With Application to SSVEP Detection," in IEEE Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2020.3027773. (https://ieeexplore.ieee.org/document/9222564)

STSVB recovers block sparse signal using matrix variate gaussian scale mixture parameterized by some scalar random parameters and 
deterministic matrices to model spatio-temporal correlation.

- stsvb_demo.m: Demo code  
- expt_SSVEP_demo.m: Demo code to run stsvb_demo.m on the dataset. For more details of dataset, please refer:
Y. Wang, X. Chen, X. Gao and S. Gao, "A Benchmark Dataset for SSVEP-Based Brain–Computer Interfaces," 
in IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 25, no. 10, pp. 1746-1752, Oct. 2017. 
Dataset is freely available on http://bci.med.tsinghua.edu.cn/download.html. 

- my_cca.m : CCA code
- myrefsig.m and genPhi.m : Auxiliary function files to run stsvb_demo.m and expt_SSVEP_demo.m

- Results.mat: File containing details of the results corresponding to Subject 27.

