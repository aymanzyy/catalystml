
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
- /𝑏𝑢𝑖𝑙𝑑: The scripts used to build the external APIs (Par-
aView, Catalyst) and the HARVEY mini-app.
- /𝑚𝑜𝑑𝑒𝑙𝑠: The model definitions for the point-cloud autoen-
coder alongside all model training and inference scripts are
included.
- /𝑟𝑢𝑛𝑠𝑐𝑟𝑖𝑝𝑡𝑠: The runscripts used to launch simulations on
Polaris are included for all experiments used to generate
data for Figures 2 and 3
- /𝑠𝑟𝑐: The HARVEY proxy-app’s source code is contained
within this directory. The core changes made to facilitate
the integration of Catalyst into the solver can be found
within the Main.cpp, CatalystAdaptor.*, and PropAB_CUDA.cpp
files. Main.cpp handles the initialization and finalization of
Catalyst instances. CatalystAdatptor.cpp is the class tasked
with data formatting as referenced in the manuscript.

--------------------------
METHODOLOGICAL INFORMATION
--------------------------

Description of methods used for collection/generation of data: 

The first set of experiments, whose results are showcased in Figure 2, encompasses the demonstration of in situ training of the point-cloud autoencoder. The experiment workflow begins with the HARVEY simulation. Fluid data is passed at runtime to train the autoencoder model. We provide the HARVEY input file \emph{proxy\_input\_file.txt}. The referenced runscripts generate simulations, with each involving a distinct training paradigm as outlined in the manuscript. Each runscript defines an input file and a bridge script as input arguments. We run the \emph{autoenc\_static\_training.sh}, \emph{autoenc\_dynamic\_training.sh}, and \emph{autoenc\_offline\_static\_training.sh} to generate the loss curves for the in situ training on a static dataset, in situ training on a dynamic dataset, and offline training on a static dataset. Model loss curves are generated with the \emph{matplotlib} library in situ and saved on disk. 


The experiment workflow began with a HARVEY simulation to generate training data for the offline training of the autoencoder model. To generate the reconstructed fluid domain illustrated in Figure 3, we then performed offline model training using the {\emph{run\_offline\_train\_for\_infer}} runscript in order to generate the weights loaded in during runtime. We then ran the 

{\emph{run\_offline\_train\_online\_infer}} runscript, simulating the HARVEY proxy app in which the train autoencoder is loaded and the encoding module is invoked to perform lossy compression. Once the simulation concludes, we evaluate the fidelity of encoding by decoding the saved latent vector using the \emph{load\_and\_decode.py} script. 

### Offline Infer, Online Train: 
- Runscripts: 
    -  runscripts/autoenc_dynamic_training.sh
    -  runscripts/autoenc_static_training.sh
    -  Whatever the other offline one was
- Input Files:
    - runscripts/proxy_input_file.txt
- Model Scripts:
    - models/autoencoder_insitu_train.py
    - models/autoencoder_insitu_train_last_call.py
    - models/offline_static_training.py
- Bridge Scripts:
    - models/autoencoder_train.py
    - models/autoencoder_train_last_call.py

### Offline Train, Online Infer: 
- Runscripts: 
    -  runscripts/run_offline_train_online_infer.sh
- Input Files:
    - runscripts/proxy_input_file.txt
- Model Scripts:
    - models/offline_train_online_infer.py
- Bridge Scripts:
    - models/catalyst_autoencoder_offline_train_online_infer.py

--------------------------
DATA-SPECIFIC INFORMATION <Create sections for EACH data file or set, as appropriate>
--------------------------

• /𝑑𝑎𝑡𝑎: The input files, including text files defining the cylindrical input geometry and model parameters, are defined.

Variable/field list
Define each including spelling out abbreviations

Value/attribute list
Include units of measure, codes or symbols used
   
Missing data treatments (null, -99, na, etc.)

-------------------------
USE and ACCESS INFORMATION 
--------------------------

Data License:

Other Rights Information: 

To cite the data: 



