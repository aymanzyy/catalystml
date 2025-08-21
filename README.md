
This readme file was generated on 08/20/25 by Ayman Yousef

Title of Dataset:
Lightweight, Modular Online Model Inference & Training in Parallel Solvers With Catalyst

Description: HARVEY Proxy App + Scripts for Instrumenting Catalyst-ParaView machine learning

Author Contact Information (Name, Institution, Email, ORCID)

	Principal Investigator: Amanda Randles 
	Institution: Duke University
	Email: amanda.randles@duke.edu
	ORCID: 0000-0001-6318-3885

	Associate or Co-investigator: Ayman Yousef
	Institution: Duke University
	Email: ayman.yousef@duke.edu
	ORCID: 0000-0002-3143-1618

--------------------
DATA & FILE OVERVIEW
--------------------

File list (filenames, directory structure (for zipped files) and brief description of all data files):

- /𝑏𝑟𝑖𝑑𝑔𝑒: The “bridge” scripts are Python scripts acting as intermediates between the solver-side and model-side. The bridge script is passed as an argument in the runscript.
- /𝑏𝑢𝑖𝑙𝑑: The scripts used to build the external APIs (ParaView, Catalyst) and the HARVEY mini-app.
- /𝑚𝑜𝑑𝑒𝑙𝑠: The model definitions for the point-cloud autoencoder alongside all model training and inference scripts are included.
- /𝑟𝑢𝑛𝑠𝑐𝑟𝑖𝑝𝑡𝑠: The runscripts used to launch simulations on Polaris are included for all experiments used to generate data for Figures 2 and 3
- /𝑠𝑟𝑐: The HARVEY proxy-app’s source code is contained within this directory. The core changes made to facilitate the integration of Catalyst into the solver can be found within the Main.cpp, CatalystAdaptor.*, and PropAB_CUDA.cpp files. Main.cpp handles the initialization and finalization of Catalyst instances. CatalystAdatptor.cpp is the class tasked with data formatting as referenced in the manuscript.

--------------------------
METHODOLOGICAL INFORMATION
--------------------------

Description of methods used for collection/generation of data: 

The first set of experiments, whose results are showcased in Figure 2, encompasses the demonstration of in situ training of the point-cloud autoencoder. The experiment workflow begins with the HARVEY simulation. Fluid data is passed at runtime to train the autoencoder model. We provide the HARVEY input file proxy_input_file.txt. The referenced runscripts generate simulations, with each involving a distinct training paradigm as outlined in the manuscript. Each runscript defines an input file and a bridge script as input arguments. We run the autoenc_static_training.sh, autoenc_dynamic_training.sh, and autoenc_offline_static_training.sh to generate the loss curves for the in situ training on a static dataset, in situ training on a dynamic dataset, and offline training on a static dataset. Model loss curves are generated with the matplotlib library in situ and saved on disk. 


The experiment workflow began with a HARVEY simulation to generate training data for the offline training of the autoencoder model. To generate the reconstructed fluid domain illustrated in Figure 3, we then performed offline model training using the run_offline_train_for_infer runscript in order to generate the weights loaded in during runtime. We then ran the run_offline_train_online_infer runscript, simulating the HARVEY proxy app in which the train autoencoder is loaded and the encoding module is invoked to perform lossy compression. Once the simulation concludes, we evaluate the fidelity of encoding by decoding the saved latent vector using the load_and_decode.py script. 

### Offline Infer, Online Train: 
- Runscripts: 
    -  runscripts/autoenc_dynamic_training.sh
    -  runscripts/autoenc_static_training.sh
    -  runscripts/autoenc_offline_static_training.sh
- Input Files:
    - data/proxy_input_file.txt
- Model Scripts:
    - models/autoencoder_insitu_train.py
    - models/autoencoder_insitu_train_last_call.py
    - models/offline_static_training.py
- Bridge Scripts:
    - bridge/autoencoder_train.py
    - bridge/autoencoder_train_last_call.py

### Offline Train, Online Infer: 
- Runscripts: 
    -  runscripts/run_offline_train_for_infer.sh
    -  runscripts/run_offline_train_online_infer.sh
- Input Files:
    - data/proxy_input_file.txt
- Model Scripts:
    - models/single_rank_autoenc.py
    - models/offline_train_online_infer.py
- Bridge Scripts:
    - bridge/autoencoder_offline_train_online_infer.py

--------------------------
DATA-SPECIFIC INFORMATION 
--------------------------

- /𝑑𝑎𝑡𝑎: The input files, including text files defining the cylindrical input geometry and model parameters, are defined.
    - cylinder.txt: Text file defining cylinder object
    - offline_trained_insitu_encoding_timestep_990.pt: Pickled latent encoding
    - proxy_input_file.txt: Input file parsed by the HARVEY proxy app

-------------------------
USE and ACCESS INFORMATION 
--------------------------

Data License:

Other Rights Information: 

To cite the data: 



