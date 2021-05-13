# Equivariant Neural Network

### Background
  some references for  group theory and representation theory.
  1. [Carter, Visual Group Theory](https://www.amazon.com/Visual-Group-Theory-Problem-Book/dp/088385757X)   
     Note: very basic intro to group theory
  2. [Theoretical Aspects of Group Equivariant Neural Networks](https://arxiv.org/abs/2004.05154)  
     Carlos Esteves  
     Note: covers all the math you need for equivariant nets in a fairly compact and accessible manner.
  3. [Serre, Linear Representations of Finite Groups](http://www.math.tau.ac.il/~borovoi/courses/ReprFG/Hatzagot.pdf)   
     Note: classic text on representations of finite groups. First few chapters are relevant to equivariant nets.
  4. [G B Folland. A Course in Abstract Harmonic Analysis](https://sv.20file.org/up1/1415_0.pdf)   
     Note: covers representations of locally compact groups; induced representations.
  5. [David Gurarie. Symmetries and Laplacians: Introduction to Harmonic Analysis, Group Representations and Applications.](https://www.amazon.com/Symmetries-Laplacians-Introduction-Representations-Applications/dp/0486462889)  
  6. [Mark Hamilton. Mathematical Gauge Theory: With Applications to the Standard Model of Particle Physics](https://www.amazon.com/Mathematical-Gauge-Theory-Applications-Universitext/dp/3319684388)   
     Note: covers fiber bundles, useful for understanding homogeneous G-CNNs and Gauge CNNs.
     
### Theory

1. **On the Generalization of Equivariance and Convolution in Neural Networks to the Action of Compact Groups**  
   Risi Kondor, Shubhendu Trivedi ICML 2018 [paper](https://arxiv.org/abs/1802.03690)  
   Note: convolution is all you need (for scalar fields)
2. **A General Theory of Equivariant CNNs on Homogeneous Spaces**  
   Taco Cohen, Mario Geiger, Maurice Weiler NeurIPS 2019 [paper](https://arxiv.org/abs/1811.02017)  
   Note: convolution is all you need (for general fields)
3. **Equivariance Through Parameter-Sharing**  
   Siamak Ravanbakhsh, Jeff Schneider, Barnabas Poczos ICML 2017 [paper](https://arxiv.org/abs/1702.08389)
4. **Universal approximations of invariant maps by neural networks**  
   Dmitry Yarotsky [paper](https://arxiv.org/abs/1804.10306)
5. **A Wigner-Eckart Theorem for Group Equivariant Convolution Kernels**  
   Leon Lang, Maurice Weiler ICLR 2021 [paper](https://arxiv.org/abs/2010.10952)  
   Note: steerable kernel spaces are fully understood and parameterized in terms of 1) generalized reduced matrix elements, 2) Clebsch-Gordan coefficients, and 3) harmonic basis functions on homogeneous spaces.
6. **On the Universality of Rotation Equivariant Point Cloud Networks**  
   Nadav Dym, Haggai Maron ICLR 2021 [paper](https://arxiv.org/abs/2010.02449),   
   Note: universality for TFN and se3-transformer 
7. **Universal Equivariant Multilayer Perceptrons**  
   Siamak Ravanbakhsh [paper](https://arxiv.org/abs/2002.02912)
8. **Provably Strict Generalisation Benefit for Equivariant Models**  
   Bryn Elesedy, Sheheryar Zaidi [paper](https://arxiv.org/abs/2102.10333)

### Application
1. **Trajectory Prediction using Equivariant Continuous Convolution**  
    Robin Walters, Jinxi Li, Rose Yu ICLR 2021 [paper](https://arxiv.org/abs/2010.11344)
2. **Incorporating Symmetry into Deep Dynamics Models for Improved Generalization**  
    Rui Wang, Robin Walters, Rose Yu ICLR 2021 [paper](https://arxiv.org/abs/2002.03061)
3. **SE(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials**  
    Simon Batzner, Tess E. Smidt, Lixin Sun, Jonathan P. Mailoa, Mordechai Kornbluth, Nicola Molinari, Boris Kozinsky [paper](https://arxiv.org/abs/2101.03164)
4. **Finding Symmetry Breaking Order Parameters with Euclidean Neural Networks**  
    Tess E. Smidt, Mario Geiger, Benjamin Kurt Miller [paper](https://arxiv.org/abs/2007.02005)
5. **Group Equivariant Generative Adversarial Networks**  
    Neel Dey, Antong Chen, Soheil Ghafurian ICLR 2021  [paper](https://arxiv.org/abs/2005.01683)   
6. **Ab-Initio Solution of the Many-Electron Schrödinger Equation with Deep Neural Networks**  
    David Pfau, James S. Spencer, Alexander G. de G. Matthews, W. M. C. Foulkes [paper](https://arxiv.org/abs/1909.02487)  
7. **Symmetry-Aware Actor-Critic for 3D Molecular Design**    
    Gregor N. C. Simm, Robert Pinsler, Gábor Csányi, José Miguel Hernández-Lobato ICLR 2021 [paper](https://arxiv.org/abs/2011.12747)
8. **Roto-translation equivariant convolutional networks: Application to histopathology image analysis**  
    Maxime W. Lafarge, Erik J. Bekkers, Josien P.W. Pluim, Remco Duits, Mitko Veta MedIA [paper](https://arxiv.org/abs/2002.08725)
9. **Scale Equivariance Improves Siamese Tracking**  
    Ivan Sosnovik\*, Artem Moskalev\*, Arnold Smeulders WACV 2021 [paper](https://arxiv.org/abs/2007.09115)
10. **3D G-CNNs for Pulmonary Nodule Detection**
    Marysia Winkels, Taco S. Cohen [paper](https://arxiv.org/abs/1804.04656) 
    International Conference on Medical Imaging with Deep Learning (MIDL), 2018.
11. **Roto-translation covariant convolutional networks for medical image analysis**  
    Erik J. Bekkers, Maxime W. Lafarge, Mitko Veta, Koen A.J. Eppenhof, Josien P.W. Pluim, Remco Duits MICCAI 2018 Young Scientist Award [paper](https://arxiv.org/abs/1804.03393)
12. **Equivariant Spherical Deconvolution: Learning Sparse Orientation Distribution Functions from Spherical Data**  
    Axel Elaldi\*, Neel Dey\*, Heejong Kim, Guido Gerig, Information Processing in Medical Imaging (IPMI) 2021 [paper](https://arxiv.org/abs/2102.09462)
13. **Rotation-Equivariant Deep Learning for Diffusion MRI**  
    Philip Müller, Vladimir Golkov, Valentina Tomassini, Daniel Cremers [paper](https://arxiv.org/abs/2102.06942)

### Tutorial

1. IAS: [Graph Nets: The Next Generation - Max Welling - YouTube](https://www.youtube.com/watch?v=Wx8J-Kw3fTA&t=3602s)
2. [Equivariance and Data Augmentation workshop](https://sites.google.com/view/equiv-data-aug/home): many nice talks
3. IPAM: [Tess Smidt: "Euclidean Neural Networks for Emulating Ab Initio Calculations and Generating Atomi..." - YouTube](https://www.youtube.com/watch?v=8CF8Grb_brE)
4. IPAM: [E(3) Equivariant Neural Network Tutorial ](https://blondegeek.github.io/e3nn_tutorial/)
5. IPAM: [Risi Kondor: "Fourier space neural networks" ](https://www.youtube.com/watch?v=-PVyi0Keiec)
6. [NeurIPS 2020 tutorial: Equivariant Networks](https://nips.cc/virtual/2020/public/tutorial_3e267ff3c8b6621e5ad4d0f26142892b.html)
7. [Yaron Lipman - Deep Learning of Irregular and Geometric Data - YouTube](https://www.youtube.com/watch?v=fveyx5zKReo&feature=youtu.be)
8. Math-ML: [Erik J Bekkers: Group Equivariant CNNs beyond Roto-Translations: B-Spline CNNs on Lie Groups](https://youtu.be/rakcnrgX4oo)
9. Kostas Daniilidis: [Geometry-aware deep learning: A brief history of equivariant representations and recent results](https://mathinstitutes.org/videos/videos/view/15146)
10. Andrew White: [ Deep Learning for Molecules and Materials.](https://whitead.github.io/dmol-book/dl/Equivariant.html)
### Equivariant

1. **Group Equivariant Convolutional Networks**  
   Taco S. Cohen, Max Welling ICML 2016 [paper](https://arxiv.org/pdf/1602.07576.pdf)   
   Note: first paper; discrete group; 
2. **Steerable CNNs**  
    Taco S. Cohen, Max Welling ICLR 2017 [paper](https://arxiv.org/abs/1612.08498)
3. **Harmonic Networks: Deep Translation and Rotation Equivariance**  
    Daniel E. Worrall, Stephan J. Garbin, Daniyar Turmukhambetov, Gabriel J. Brostow CVPR 2017 [paper](https://arxiv.org/abs/1612.04642)   
4. **Spherical CNNs**  
    Taco S. Cohen, Mario Geiger, Jonas Koehler, Max Welling ICLR 2018 best paper  [paper](https://arxiv.org/abs/1801.10130)  
    Note: use generalized FFT to speed up convolution on $S^2$ and $SO(3)$
5. **Clebsch–Gordan Nets: a Fully Fourier Space Spherical Convolutional Neural Network**  
    Risi Kondor, Zhen Lin, Shubhendu Trivedi NeurIPS 2018 [paper](https://arxiv.org/abs/1806.09231)  
    Note: perform equivariant nonlinearity in Fourier space; 
6. **General E(2)-Equivariant Steerable CNNs**  
    Maurice Weiler, Gabriele Cesa NeurIPS 2019 [paper](https://arxiv.org/abs/1911.08251)  
    Note: nice benchmark on different reprsentations
7. **Learning Steerable Filters for Rotation Equivariant CNNs**  
   Maurice Weiler, Fred A. Hamprecht, Martin Storath CVPR 2018 [paper](https://arxiv.org/abs/1711.07289)   
   Note: group convolutions, kernels parameterized in circular harmonic basis (steerable filters);
8. **Learning SO(3) Equivariant Representations with Spherical CNNs**  
   Carlos Esteves, Christine Allen-Blanchette, Ameesh Makadia, Kostas Daniilidis ECCV 2018 [paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Carlos_Esteves_Learning_SO3_Equivariant_ECCV_2018_paper.html)  
   Note: SO(3) equivariance; zonal filter
9. **Polar Transformer Networks**  
    Carlos Esteves, Christine Allen-Blanchette, Xiaowei Zhou, Kostas Daniilidis ICLR 2018 [paper](https://arxiv.org/abs/1709.01889)  
10. **3D Steerable CNNs: Learning Rotationally Equivariant Features in Volumetric Data**  
    Maurice Weiler, Mario Geiger, Max Welling, Wouter Boomsma, Taco Cohen  NeurIPS 2018 [paper](https://arxiv.org/abs/1807.02547)  
    Note: SE(3) equivariance; characterize the basis of steerable kernel
11. **Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds**  
      Nathaniel Thomas, Tess Smidt, Steven Kearnes, Lusann Yang, Li Li, Kai Kohlhoff, Patrick Riley  [paper](https://arxiv.org/abs/1802.08219)  
      Note: SE(3) equivariance for point clouds
12. **Equivariant Multi-View Networks**  
      Carlos Esteves, Yinshuang Xu, Christine Allen-Blanchette, Kostas Daniilidis  ICCV 2019 [paper](https://arxiv.org/abs/1904.00993)   
13. **Gauge Equivariant Convolutional Networks and the Icosahedral CNN**  
      Taco S. Cohen, Maurice Weiler, Berkay Kicanaoglu, Max Welling ICML 2019 [paper](https://arxiv.org/abs/1902.04615), [talk](https://slideslive.com/38915809/gauge-equivariant-convolutional-networks?locale=de)  
      Note: gauge equivariance on general manifold
14. **Cormorant: Covariant Molecular Neural Networks**  
      Brandon Anderson, Truong-Son Hy, Risi Kondor NeurIPS 2019 [paper](https://arxiv.org/abs/1906.04015)
15. **Deep Scale-spaces: Equivariance Over Scale**  
      Daniel Worrall, Max Welling NeurIPS 2019 [paper](https://papers.nips.cc/paper/2019/hash/f04cd7399b2b0128970efb6d20b5c551-Abstract.html)
16. **Scale-Equivariant Steerable Networks**  
      Ivan Sosnovik, Michał Szmaja, Arnold Smeulders ICLR 2020 [paper](https://openreview.net/forum?id=HJgpugrKPS)
17. **B-Spline CNNs on Lie Groups**  
      Erik J Bekkers ICLR 2020 [paper](https://openreview.net/forum?id=H1gBhkBFDH)    
18. **SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks**  
      Fabian B. Fuchs, Daniel E. Worrall, Volker Fischer, Max Welling NeurIPS 2020  [paper](https://arxiv.org/abs/2006.10503), [blog](https://fabianfuchsml.github.io/se3transformer/)  
      Note: TFN + equivariant self-attention; improved spherical harmonics computation
19. **Gauge Equivariant Mesh CNNs: Anisotropic convolutions on geometric graphs**  
      Pim de Haan, Maurice Weiler, Taco Cohen, Max Welling ICLR 2021 [paper](https://arxiv.org/abs/2003.05425)  
      Note: anisotropic gauge equivariant kernels + message passing  by parallel transporting features over mesh edges
20. **Lorentz Group Equivariant Neural Network for Particle Physics**  
      Alexander Bogatskiy, Brandon Anderson, Jan T. Offermann, Marwah Roussi, David W. Miller, Risi Kondor ICML 2020 [paper](https://arxiv.org/abs/2006.04780)  
      Note: SO(1, 3) equivariance
21. **Generalizing Convolutional Neural Networks for Equivariance to Lie Groups on Arbitrary Continuous Data**  
      Marc Finzi, Samuel Stanton, Pavel Izmailov, Andrew Gordon Wilson ICML 2020 [paper](https://arxiv.org/abs/2002.12880)  
      Note: fairly generic architecture; use Monte Carlo sampling to achieve equivariance in expectation; 
22. **Spin-Weighted Spherical CNNs**  
      Carlos Esteves, Ameesh Makadia, Kostas Daniilidis NeurIPS 2020 [paper](https://arxiv.org/abs/2006.10731)  
      Note: anisotropic filter for vector field on sphere
23. **Learning Invariances in Neural Networks**  
      Gregory Benton, Marc Finzi, Pavel Izmailov, Andrew Gordon Wilson NeurIPS 2020 [paper](https://arxiv.org/abs/2010.11882)   
      Note: very interesting approch; enfore "soft" invariance via learning over both model parameters and distributions over augmentations
24. **Lie Algebra Convolutional Neural Networks with Automatic Symmetry Extraction**  
      Nima Dehmamy, Yanchen Liu, Robin Walters, Rose Yu  [paper](https://openreview.net/forum?id=cTQnZPLIohy)    
      Note: very interesting paper; It’s unfortunate that it is rejected by ICLR 2021  
25. **LieTransformer: Equivariant self-attention for Lie Groups**  
      Michael Hutchinson, Charline Le Lan, Sheheryar Zaidi, Emilien Dupont, Yee Whye Teh, Hyunjik Kim [paper](https://arxiv.org/abs/2012.10885)  
      Note: equivariant self attention to arbitrary Lie groups and their discrete subgroups
26. **Co-Attentive Equivariant Neural Networks: Focusing Equivariance On Transformations Co-Occurring In Data**  
      David W. Romero, Mark Hoogendoorn ICLR 2020 [paper](https://arxiv.org/abs/1911.07849)
27. **Attentive Group Equivariant Convolutional Networks**  
      David W. Romero, Erik J. Bekkers, Jakub M. Tomczak, Mark Hoogendoorn ICML 2020 [paper](https://arxiv.org/abs/2002.03830)
28. **Wavelet Networks: Scale Equivariant Learning From Raw Waveforms**  
      David W. Romero, Erik J. Bekkers, Jakub M. Tomczak, Mark Hoogendoorn [paper](https://arxiv.org/abs/2006.05259)
29. **Group Equivariant Stand-Alone Self-Attention For Vision**  
      David W. Romero, Jean-Baptiste Cordonnier ICLR 2021 [paper](https://arxiv.org/abs/2010.00977)
30. **MDP Homomorphic Networks: Group Symmetries in Reinforcement Learning**  
      Elise van der Pol, Daniel E. Worrall, Herke van Hoof, Frans A. Oliehoek, Max Welling NeurIPS 2020 [paper](https://arxiv.org/abs/2006.16908)
31. **Isometric Transformation Invariant and Equivariant Graph Convolutional Networks**  
      Masanobu Horie, Naoki Morita, Toshiaki Hishinuma, Yu Ihara, Naoto Mitsume ICLR 2021 [paper](https://arxiv.org/abs/2005.06316)
32. **Making Convolutional Networks Shift-Invariant Again**   
     Richard Zhang ICML 2019 [paper](https://arxiv.org/abs/1904.11486)
33. **Probabilistic symmetries and invariant neural networks**  
     Benjamin Bloem-Reddy, Yee Whye Teh JMLR [paper](https://arxiv.org/abs/1901.06082)
34. **On Representing (Anti)Symmetric Functions**  
     Marcus Hutter [paper](https://arxiv.org/abs/2007.15298)
35. **PDE-based Group Equivariant Convolutional Neural Networks**  
     Bart M.N. Smets, Jim Portegies, Erik J. Bekkers, Remco Duits [paper](https://arxiv.org/abs/2001.09046)
36. **PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation**  
     Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas CVPR 2017 [paper](https://arxiv.org/abs/1612.00593) 
37. **Deep Sets**  
     Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan Salakhutdinov, Alexander Smola NeurIPS 2017   [paper](https://arxiv.org/abs/1703.06114)
38. **Invariant and Equivariant Graph Networks**  
     Haggai Maron, Heli Ben-Hamu, Nadav Shamir, Yaron Lipman ICLR 2019 [paper](https://arxiv.org/abs/1812.09902)  
39. **Provably Powerful Graph Networks**  
     Haggai Maron, Heli Ben-Hamu, Hadar Serviansky, Yaron Lipman NeurIPS 2019 [paper](https://arxiv.org/abs/1905.11136)  
40. **Universal Invariant and Equivariant Graph Neural Networks**  
     Nicolas Keriven, Gabriel Peyré NeurIPS 2019 [paper](https://papers.nips.cc/paper/2019/hash/ea9268cb43f55d1d12380fb6ea5bf572-Abstract.html)
41. **On Learning Sets of Symmetric Elements**  
     Haggai Maron, Or Litany, Gal Chechik, Ethan Fetaya [ICML 2020 best paper](https://arxiv.org/abs/2002.08599)
42. **On the Universality of Invariant Networks**  
     Haggai Maron, Ethan Fetaya, Nimrod Segol, Yaron Lipman [paper](https://arxiv.org/abs/1901.09342)
43. **Equivariant Flows: Exact Likelihood Generative Learning for Symmetric Densities**    
     Jonas Köhler, Leon Klein, Frank Noé ICML 2020 [paper](https://arxiv.org/abs/2006.02425)  
     Note: general framework for constructing equivariant normalizing flows on euclidean spaces. Instantiation for particle systems/point clouds = simultanoues SE(3) and permutation equivariance.
44. **Equivariant Hamiltonian Flows**    
     Danilo Jimenez Rezende, Sébastien Racanière, Irina Higgins, Peter Toth NeurIPS 2019 ML4Phys workshop [paper](https://arxiv.org/abs/1909.13739)  
     Note: general framework for constructing equivariant normalizing flows in phase space utilizing Hamiltonian dynamics. Instantiation for SE(2) equivariance.
45. **Sampling using SU(N) gauge equivariant flows**    
     Denis Boyda, Gurtej Kanwar, Sébastien Racanière, Danilo Jimenez Rezende, Michael S. Albergo, Kyle Cranmer, Daniel C. Hackett, Phiala E. Shanahan [paper](https://arxiv.org/abs/2008.05456)    
     Note: normalizing flows for lattice gauge theory. Instantiation for SU(2)/SU(3) equivariance.
46. **Exchangeable neural ode for set modeling**    
     Yang Li, Haidong Yi, Christopher M. Bender, Siyuan Shan, Junier B. Oliva NeurIPS 2020 [paper](https://arxiv.org/abs/2008.02676)  
     Note: framework for permutation equivariant flows for set data. Instantiation for permutation equivariance.
47. **Equivariant Normalizing Flows for Point Processes and Sets**    
     Marin Biloš, Stephan Günnemann NeurIPS 2020 [paper](https://arxiv.org/abs/2010.03242)  
     Note: framework for permutation equivariant flows for set data.  Instantiation for permutation equivariance.
48. **The Convolution Exponential and Generalized Sylvester Flows**    
     Emiel Hoogeboom, Victor Garcia Satorras, Jakub M. Tomczak, Max Welling NeurIPS 2020 [paper](https://arxiv.org/abs/2006.01910)  
     Note: invertible convolution operators. Instantiation for permutation equivariance.
49. **Targeted free energy estimation via learned mappings**    
     Peter Wirnsberger, Andrew J. Ballard, George Papamakarios, Stuart Abercrombie, Sébastien Racanière, Alexander Pritzel, Danilo Jimenez Rezende, Charles Blundell J Chem Phys. 2020 Oct 14;153(14):144112. [paper](https://arxiv.org/abs/2002.04913)  
     Note: normalizing flows for particle systems on a torus. Instantiation for permutation equivariance.
50. **Temperature-steerable flows**    
     Manuel Dibak, Leon Klein, Frank Noé NeurIPS 2020 ML4Phys workshops [paper](https://arxiv.org/abs/2012.00429)  
     Note: normalizing flows in phase space with equivariance with respect to changes in temperature.
     
###  Codes

1. dynamics prediction using permutation-equivariant neural networks https://github.com/arayabrain/PermutationalNetworks

2. Deeply Supervised Rotation Equivariant Network for Lesion Segmentation in Dermoscopy Images https://github.com/xmengli999/Deeply-Supervised-Rotation-Equivariant-Network-for-Lesion-Segmentation

3. An Equivariant Bayesian Convolutional Network predicts recombination hotspots and accurately resolves binding motifs https://github.com/luntergroup/EquivariantNetworks

4. Group Equivariant Convolutional Networks https://github.com/tscohen/gconv_experiments

5. Universal attacks on equivariant networks https://github.com/smerdov/Universal-attacks-on-equivariant-networks

6. Group Equivariant Capsule Networks https://github.com/mrjel/group_equivariant_capsules_pytorch

7. Rotation *equivariant* vector field *networks* (ICCV 2017) PyTorch  https://github.com/COGMAR/RotEqNet

8. Rotation *equivariant* vector field *networks* (ICCV 2017) https://github.com/dmarcosg/RotEqNet

9. Equivariant neural networks and equivarification https://github.com/symplecticgeometry/equivariant-neural-networks-and-equivarification

10. Rotation *Equivariant* Deep Neural *Network* https://github.com/red-nn/RED-NN

11. Rotation-equivariant convolutional neural network ensembles in image processing https://github.com/LouiseHash/Rotation_Equivariant_CNN_Ensembles

12. Universal Invariant and Equivariant Graph Neural Networks  (NeurIPS 2019) https://github.com/nkeriven/univgnn

13. On Universal Equivariant Set Networks,  ICLR 2020  https://github.com/NimrodSegol/On-Universal-Equivariant-Set-Networks

14. Multiple Sequence Imputation with Equivariant Dual Graph Networks https://github.com/YudeWang/SSENet-pytorch

15. self-supervised scale equivariant network for weakly supervised semantic segmentation pytorch https://github.com/CS4240-Group67/Group-Equivariant-Convolutional-Networks

16. Group Equivariant Convolutional Networks  Jupyter Notebook https://github.com/facebookresearch/Permutation-Equivariant-Seq2Seq

17. Finite Group equivariant Neural Networks https://github.com/FGNN-Author/FGNN

18. Learning irreducible representations of noncommutative Lie groups, applied to constructing group equivariant neural networks. https://github.com/noajshu/learning_irreps

19. TensorFlow implementation of Equivariant Transformer Networks https://github.com/julianroth/equivariant-transformers

20. $SO(3)$-equivariant quaternion convolutional kernel + Convolutional Neural Networks for accelerometric gait https://github.com/vinayprabhu/QNN_gait

21. Attentive Group Equivariant Convolutional Neural Networks" published at ICML 2020 https://github.com/dwromero/att_gconvs

22. Wavelet Networks: Scale Equivariant Learning From Raw Waveforms https://github.com/dwromero/wavelet_networks

23. Quaternion Equivariant Capsule Networks for 3D Point Clouds https://github.com/tolgabirdal/qenetworks

24. Rotation Equivariant Graph Convolutional Network for Spherical Image Classification https://github.com/QinYang12/SGCN

25. RotDCF: Decomposition of Convolutional Filters for Rotation Equivariant Deep Networks https://github.com/ZichenMiao/RotDCF

26. Equivariant Neural Network Molecular Dynamics https://github.com/rajak7/ML_MD

27. Scale-Equivariant Steerable Networks https://github.com/ISosnovik/sesn

28. 3D-Rotation-Equivariant Quaternion Neural Networks  https://github.com/ada-shen/REQNN

29. Lorentz Group Equivariant Neural Network for Particle Physics, ICML 2020 https://github.com/fizisist/LorentzGroupNetwork

29. 3D Rotation / Translation Equivariant Attention Networks, in Pytorch https://github.com/zeta1999/SE3-Transformers

30. Relevance of Rotationally Equivariant Convolutions for Predicting Molecular Properties https://github.com/bkmi/equivariant-benchmark

31. Quaternion Equivariant Capsule Networks for 3D Point Clouds https://github.com/qq456cvb/QENet

32. Attentive Group Equivariant Convolutional Neural Networks ,  ICML 2020. https://github.com/h-roy/Attentive-Group-Equivariant-Convolutional-Networks

33. Equivariant Seq2Seq Network https://github.com/danielTLevy/EquivSeq2Seq

34. Rotation Equivariant Siamese Networks for Tracking https://github.com/dkgupta90/re-siamnet

35. Rotationally and translationally equivariant layers and networks for deep learning on diffusion MRI scans https://github.com/philip-mueller/equivariant-deep-dmri

36. Fanaroff-Riley classification of radio galaxies using group-equivariant convolutional neural networks, 2021  https://github.com/as595/E2CNNRadGal

37. tutorials on 3d Euclidean equivariant neural networks https://github.com/blondegeek/e3nn_tutorial

38. Equivariant Energy Flow Networks for jet tagging https://github.com/ayo-ore/equivariant-efns

39. Equivariant Multi-View Networks https://github.com/daniilidis-group/emvn

40. Quaternion Equivariant Capsule Networks for 3D Point Clouds https://github.com/tolgabirdal/qecnetworks

41. Group Equivariant Convolutional Neural Networks https://github.com/tscohen/GrouPy

42. Equivariant Point Network for 3D Point Cloud Analysis (CVPR2021) https://github.com/nintendops/EPN_PointCloud

43. An implementation of Equivariant e2 convolutional kernals into a convolutional self attention network, applied to radio astronomy data. https://github.com/mb010/EquivariantSelfAttention

44. Vector Neurons: A General Framework for SO(3)-Equivariant Networks https://github.com/FlyingGiraffe/vnn

45. Group Equivariant Generative Adversarial Networks, ICLR 2021. https://github.com/neel-dey/equivariant-gans

46. Scale Equivariant Neural Networks with Morphological Scale-Spaces https://github.com/mateussangalli/morphological-scale-space-networks

47. E(n)-Equivariant Graph Neural Networks, in Pytorch https://github.com/lucidrains/egnn-pytorch

48. E(n)-Equivariant Transformer https://github.com/lucidrains/En-transformer

### Papers                     
<ul>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(1).pdf" style="text-decoration:none;">A Course in Abstract Harmonic Analysis</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(2).pdf" style="text-decoration:none;">Group Equivariant Convolutional Networks</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(3).pdf" style="text-decoration:none;">PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(4).pdf" style="text-decoration:none;">Harmonic Networks: Deep Translation and Rotation Equivariance</a></li>      
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(5).pdf" style="text-decoration:none;">Steerable CNNs</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(6).pdf" style="text-decoration:none;">Equivariance Through Parameter-Sharing</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(7).pdf" style="text-decoration:none;">Deep Sets</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(8).pdf" style="text-decoration:none;"> Polar Transformer Networks </a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(9).pdf" style="text-decoration:none;">Learning Steerable Filters for Rotation Equivariant CNNs</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(10).pdf" style="text-decoration:none;">Spherical CNNs </a></li>  
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(11).pdf" style="text-decoration:none;">On the Generalization of Equivariance and Convolution in Neural Networks to the Action of Compact Groups</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(12).pdf" style="text-decoration:none;">Tensor field networks:Rotation- and translation-equivariant neural networks for 3D point clouds</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(13).pdf" style="text-decoration:none;">Roto-Translation Covariant Convolutional Networks for Medical Image Analysis</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(14).pdf" style="text-decoration:none;">3D G-CNNs for Pulmonary Nodule Detection</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(15).pdf" style="text-decoration:none;">Universal approximations of invariant maps by neural networks</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(16).pdf" style="text-decoration:none;">Clebsch–Gordan Nets: a Fully Fourier Space Spherical Convolutional Neural Network</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(17).pdf" style="text-decoration:none;">3D Steerable CNNs: Learning Rotationally Equivariant Features in Volumetric Data</a></li>   
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(18).pdf" style="text-decoration:none;">A General Theory of Equivariant CNNs on Homogeneous Spaces</a></li> 
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(19).pdf" style="text-decoration:none;">Invariant and Equivariant Graph Networks</a></li> 
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(20).pdf" style="text-decoration:none;">On the Universality of Invariant Networks</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(21).pdf" style="text-decoration:none;">Gauge Equivariant Convolutional Networks and the Icosahedral CNN</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(22).pdf" style="text-decoration:none;">Equivariant Multi-View Networks</a></li> 
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(23).pdf" style="text-decoration:none;">Provably Powerful Graph Networks</a></li> 
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(24).pdf" style="text-decoration:none;">Cormorant: Covariant Molecular Neural Networks</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(25).pdf" style="text-decoration:none;">Ab-Initio Solution of the Many-Electron Schrodinger Equation with Deep Neural Networks</a></li>  
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(26).pdf" style="text-decoration:none;">Equivariant Hamiltonian Flows</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(27).pdf" style="text-decoration:none;">Co-Attentive Equivariant Neural Networks: Focusing Equivariance On Transformations Co-Occurring In Data</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(28).pdf" style="text-decoration:none;">General E(2) - Equivariant Steerable CNNs</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(29).pdf" style="text-decoration:none;">Universal Equivariant Multilayer Perceptrons </a></li>                              
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(30).pdf" style="text-decoration:none;">Incorporating Symmetry into Deep Dynamics Models for Improved Generalization</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(31).pdf" style="text-decoration:none;">Attentive Group Equivariant Convolutional Networks</a></li> 
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(32).pdf" style="text-decoration:none;">Targeted free energy estimation via learned mappings</a></li> 
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(33).pdf" style="text-decoration:none;">On Learning Sets of Symmetric Elements</a></li>               
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(34).pdf" style="text-decoration:none;">Roto-Translation Equivariant Convolutional Networks: Application to Histopathology Image Analysis</a></li> 
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(35).pdf" style="text-decoration:none;">Generalizing Convolutional Neural Networks for Equivariance to Lie Groups on Arbitrary Continuous Data</a></li> 
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(36).pdf" style="text-decoration:none;">Gauge Equivariant Mesh CNNs Anisotropic convolutions on geometric graphs</a></li> 
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(37).pdf" style="text-decoration:none;">Isometric Transformation Invariant and Equivariant Graph Convolutional Networks</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(38).pdf" style="text-decoration:none;">The Convolution Exponential and Generalized Sylvester Flows</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(39).pdf" style="text-decoration:none;">Equivariant Flows: Exact Likelihood Generative Learning for Symmetric Densities</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(40).pdf" style="text-decoration:none;">Lorentz Group Equivariant Neural Network for Particle Physics</a></li> 
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(41).pdf" style="text-decoration:none;">Wavelet Networks:Scale Equivariant Learning From Raw Waveforms</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(42).pdf" style="text-decoration:none;">SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(43).pdf" style="text-decoration:none;">Spin-Weighted Spherical CNNs</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(44).pdf" style="text-decoration:none;">MDP Homomorphic Networks: Group Symmetries in Reinforcement Learning</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(45).pdf" style="text-decoration:none;">Finding symmetry-breaking Order Parameters with Euclidean Neural Networks</a></li>  
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(46).pdf" style="text-decoration:none;">Scale Equivariance Improves Siamese Tracking</a></li> 
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(47).pdf" style="text-decoration:none;">Exchangeable Neural ODE for Set Modeling</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(48).pdf" style="text-decoration:none;">Sampling using SU(N) gauge equivariant flows</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(49).pdf" style="text-decoration:none;">Group Equivariant Stand-Alone Self-Attention For Vision</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(50).pdf" style="text-decoration:none;">On the Universality of Rotation Equivariant Point Cloud Networks</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(51).pdf" style="text-decoration:none;">Equivariant Normalizing Flows for Point Processes and Sets</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(52).pdf" style="text-decoration:none;">A Wigner-Eckart Theorem for Group Equivariant Convolution Kernels</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(53).pdf" style="text-decoration:none;">Trajectory Prediction using Equivariant Continuous Convolution</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(54).pdf" style="text-decoration:none;">Learning Invariances in Neural Networks </a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(55).pdf" style="text-decoration:none;">Symmetry-Aware Actor-Critic for 3D Molecular Design</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(56).pdf" style="text-decoration:none;">Temperature-steerable flows </a></li>                              
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(57).pdf" style="text-decoration:none;">LieTransformer: Equivariant Self-Attention for Lie Groups</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(58).pdf" style="text-decoration:none;">SE(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(59).pdf" style="text-decoration:none;">Rotation-Equivariant Deep Learning for Diffusion MRI</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(60).pdf" style="text-decoration:none;">Equivariant Spherical Deconvolution: Learning Sparse Orientation Distribution Functions from Spherical Data </a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(61).pdf" style="text-decoration:none;">Provably Strict Generalisation Benefit for Equivariant Models</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(62).pdf" style="text-decoration:none;">B-Spline CNNs on Lie Groups</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(63).pdf" style="text-decoration:none;">Learning SO(3) Equivariant Representations with Spherical CNNs</a></li>      
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(64).pdf" style="text-decoration:none;">Linear· Representations of Finite Groups</a></li>
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(65).pdf" style="text-decoration:none;">Lie Algebra Convolutional Neural Networks with Automatic Symmetry Extraction </a></li> 
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(66).pdf" style="text-decoration:none;">Deep Scale-spaces: Equivariance Over Scale</a></li> 
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(67).pdf" style="text-decoration:none;">Universal Invariant and Equivariant Graph Neural Networks</a></li> 
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(68).pdf" style="text-decoration:none;">Scale-Equivariant Steerable Networks</a></li> 
<li><a target="_blank" href="https://github.com/Chunyan-Law/Geometric-Deep-Learning-Grids-Groups-Graphs-Geodesics-and-Gauges/Equivariant/e(69).pdf" style="text-decoration:none;">Group Equivariant Generative Adversarial Networks</a></li>  
 </ul>
