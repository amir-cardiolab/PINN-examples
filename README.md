# PINN-examples
Examples implementing physics-informed neural networks (PINN) in Pytorch

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \
Pytorch_NN_example: 

Linear and nonlinear regression examples with a neural network implemented in Pytorch.

An excellent detailed intro to neural networks with Pytorch:
https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \
Pytorch_PINN: 

1d_advdif_PINN.py:  Solve steady 1D advection-diffusion equation using PINN. 

2d_diffusion_PINN.py: Solve steady 2D diffusion equation with a source term using PINN

stenosis_NS.py: Solve steady 2D Navier-Stokes equation in an idealized stenosis model using PINN.
The data needed for the 2D stenosis model are located here:

https://github.com/amir-cardiolab/PINN-wss/tree/main/Data/2D-stenosis

Need to install visualization toolkint (vtk) libraries to read the input data:

conda activate pytorch \
pip install vtk

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \
Inverse modeling using PINN: 

See:
https://github.com/amir-cardiolab/PINN-wss

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \
Useful seminars about PINNs:

Karniadakis (PINN overview):\
https://www.youtube.com/watch?v=FQ0vsqU-K00&list=PLw74xLHy0_j8DXxAKb15DbgtNvUOeTPbZ&index=2&ab_channel=MSML2020Conference

Lu (solving PDEs with PINN):\
https://www.youtube.com/watch?v=Wfgr1pMA9fY&list=PLw74xLHy0_j8DXxAKb15DbgtNvUOeTPbZ&index=4&t=1879s&ab_channel=MLPS-CombiningAIandMLwithPhysicsSciences

Karniadakis (PINN overview and various applications):\
https://www.youtube.com/watch?v=7kCq2uQmQU4&list=PLw74xLHy0_j8DXxAKb15DbgtNvUOeTPbZ&index=27&t=3s&ab_channel=CambridgeUniversityPress-Academic

Raissi (inverse modeling with PINN;  SECOND HALF OF THE TALK):\
https://www.youtube.com/watch?v=iy4PIeW91_I&t=2363s&ab_channel=MLPS-CombiningAIandMLwithPhysicsSciences

Perdikaris (PINN overview and challenges):\
https://www.youtube.com/watch?v=68MtA0L9ZAI&t=3327s&ab_channel=TexasA%26MInstituteofDataScience

Wang (superresolution with PINN):\
https://www.youtube.com/watch?v=xMimSG4NBT0&t=1s&ab_channel=JianxunWang

Arzani (identifyng wall shear stress from sparse data with PINN):\
https://www.youtube.com/watch?v=rK-Bb6-0svs&ab_channel=AmirhosseinArzani





