File breakdown per experiment

Offline Infer, Online Train: 
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

Offline Train, Online Infer: 
- Runscripts: 
    -  runscripts/run_offline_train_online_infer.sh
- Input Files:
    - runscripts/proxy_input_file.txt
- Model Scripts:
    - models/offline_train_online_infer.py
- Bridge Scripts:
    - models/catalyst_autoencoder_offline_train_online_infer.py

