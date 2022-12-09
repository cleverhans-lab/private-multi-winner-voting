# Private Multi-Winner Voting for Machine Learning

### Installation Instructions:

The `environment.yml` file contains the dependencies for this repository. To install these in a new environment, run 

`conda env create -f environment.yml` followed by 

`conda activate private-multi` to activate the environment. The code has been tested on a machine running Ubuntu 20.04 LTS with 4 NVIDIA GeForce RTX 2080 graphics cards. Conda version 4.12.0 was used to install the environment. 


### Description of the code: 

The `main.py` file contains the starting point for running different experiments. The parameters for the program
are in `parameters.py` file. Several helper functions and utilities are contained in `utils.py`.

### Examples of the pipeline:

1. Train private models (from scratch):

Make sure to download the `CheXpert-v1.0-small` dataset to the `~/data` folder. The dataset can be found, for example, here: `https://www.kaggle.com/datasets/ashery/chexpert`.

```
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
DATASET='cxpert'
architecture='densenet121_cxpert'
num_models=50
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--path ~/code/private-multi-winner-voting \
--data_dir ~/data \
--dataset ${DATASET} \
--dataset_type 'balanced' \
--balance_type 'standard' \
--begin_id 0 \
--end_id ${num_models} \
--num_querying_parties 3 \
--num_models ${num_models} \
--num_epochs 100 \
--architectures ${architecture} \
--commands 'train_private_models' \
--threshold 50 \
--sigma_gnmax 7.0 \
--sigma_threshold 30 \
--budgets 20.0 \
--mode 'random' \
--lr 0.001 \
--weight_decay 0.00001 \
--schedule_factor 0.1 \
--scheduler_type 'ReduceLROnPlateau' \
--loss_type 'BCEWithLogits' \
--num_workers 8 \
--batch_size 64 \
--eval_batch_size 64 \
--class_type 'multilabel' \
--device_ids 0 1 2 3 \
--momentum 0.9 \
--weak_classes '' \
--chexpert_dataset_type 'pos' \
--log_every_epoch 0 \
--debug 'False' \
--xray_views 'AP' 'PA' \
>> train_private_${DATASET}_${timestamp}.txt 2>&1 &
echo train_private_${DATASET}_${timestamp}.txt
```

The output location of the private models is determined by the `--path` argument as well as other settings as presented in the `main.py` code:

```
args.private_model_path = os.path.join(
                    args.path,
                    "private-models",
                    dataset,
                    architecture,
                    "{:d}-models".format(args.num_models),
                    xray_views,
                )
```

The training of the private models is taking most of the time for the pipeline and depends on the selected dataset as well as model architecture. For example, on 4 NVIDIA GeForce RTX 2080 GPUs, a single epoch takes around 15 sec for the CheXpert dataset trained on DenseNet121. In the above examples, we train 50 models, each with 100 epochs. Thus, if the models are trained sequentially, then it takes around 21 hours. However, the script can be run in parallel on many machines to decrease the total training time.

2. Test private models:

```
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
DATASET='cxpert'
architecture="densenet121_${DATASET}"
num_models=50
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. nohup python main.py \
--path ~/code/private-multi-winner-voting \
--data_dir ~/data \
--dataset ${DATASET} \
--dataset_type 'balanced' \
--balance_type 'standard' \
--begin_id 0 \
--end_id ${num_models} \
--num_querying_parties 3 \
--num_models ${num_models} \
--num_epochs 100 \
--architectures ${architecture} \
--commands 'test_models' \
--sigma_gnmax 7.0 \
--threshold 50 \
--sigma_threshold 30 \
--budgets 20.0 \
--optimizer 'Adam' \
--mode 'random' \
--lr 0.001 \
--weight_decay 0.00001 \
--schedule_factor 0.1 \
--scheduler_type 'ReduceLROnPlateau' \
--loss_type 'BCEWithLogits' \
--num_workers 8 \
--batch_size 64 \
--eval_batch_size 64 \
--class_type 'multilabel' \
--device_ids 0 1 2 3 \
--momentum 0.9 \
--weak_classes '' \
--chexpert_dataset_type 'pos' \
--log_every_epoch 0 \
--debug 'False' \
--xray_views 'AP' 'PA' \
--query_set_type 'numpy' \
--pick_labels -1 \
--retrain_model_type 'load' \
--test_models_type 'private' \
>>train_private_${DATASET}_${timestamp}.txt 2>&1 &
echo train_private_${DATASET}_${timestamp}.txt
```

3. Generate query-answer pairs for re-training.

```
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
DATASET='cxpert'
architecture="densenet121_cxpert"
num_models=50
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. nohup python main.py \
--path ~/code/private-multi-winner-voting \
--data_dir ~/data \
--dataset ${DATASET} \
--dataset_type 'balanced' \
--balance_type 'standard' \
--begin_id 0 \
--end_id 3 \
--num_querying_parties 3 \
--querying_party_ids 0 1 2 \
--num_models ${num_models} \
--num_epochs 100 \
--architectures ${architecture} \
--commands 'query_ensemble_model' \
--sigma_gnmax 7.0 \
--threshold 50 \
--sigma_threshold 30 \
--budgets 20.0 \
--optimizer 'Adam' \
--mode 'random' \
--lr 0.001 \
--weight_decay 0.00001 \
--schedule_factor 0.1 \
--scheduler_type 'ReduceLROnPlateau' \
--loss_type 'BCEWithLogits' \
--num_workers 8 \
--batch_size 64 \
--eval_batch_size 1024 \
--class_type 'multilabel' \
--device_ids 0 1 2 3 \
--momentum 0.9 \
--weak_classes '' \
--chexpert_dataset_type 'pos' \
--log_every_epoch 0 \
--debug 'False' \
--xray_views 'AP' 'PA' \
--query_set_type 'numpy' \
--pick_labels -1 \
--transfer_type '' \
>>train_private_${DATASET}_${timestamp}.txt 2>&1 &
echo train_private_${DATASET}_${timestamp}.txt
```

4. Re-train private models.

```
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
DATASET='cxpert'
architecture="densenet121_${DATASET}"
num_models=50
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. nohup python main.py \
--path ~/code/private-multi-winner-voting \
--data_dir ~/data \
--dataset ${DATASET} \
--dataset_type 'balanced' \
--balance_type 'standard' \
--begin_id 0 \
--end_id 3 \
--num_querying_parties 3 \
--num_models ${num_models} \
--num_epochs 100 \
--architectures ${architecture} \
--commands 'retrain_private_models' \
--sigma_gnmax 7.0 \
--threshold 50 \
--sigma_threshold 30 \
--budgets 20.0 \
--optimizer 'Adam' \
--mode 'random' \
--lr 0.001 \
--weight_decay 0.00001 \
--schedule_factor 0.1 \
--scheduler_type 'ReduceLROnPlateau' \
--loss_type 'BCEWithLogits' \
--num_workers 8 \
--batch_size 64 \
--eval_batch_size 64 \
--class_type 'multilabel' \
--device_ids 0 1 2 3 \
--momentum 0.9 \
--weak_classes '' \
--chexpert_dataset_type 'pos' \
--log_every_epoch 0 \
--debug 'False' \
--xray_views 'AP' 'PA' \
--query_set_type 'numpy' \
--pick_labels -1 \
--retrain_model_type 'load' \
--test_models_type 'retrained' \
>>train_private_${DATASET}_${timestamp}.txt 2>&1 &
echo train_private_${DATASET}_${timestamp}.txt
```

5. Test re-trained models.

```
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
DATASET='cxpert'
architecture="densenet121_${DATASET}"
num_models=50
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. nohup python main.py \
--path ~/code/private-multi-winner-voting \
--data_dir ~/data \
--dataset ${DATASET} \
--dataset_type 'balanced' \
--balance_type 'standard' \
--begin_id 0 \
--end_id 3 \
--num_querying_parties 3 \
--num_models ${num_models} \
--num_epochs 100 \
--architectures ${architecture} \
--commands 'test_models' \
--sigma_gnmax 7.0 \
--threshold 50 \
--sigma_threshold 30 \
--budgets 20.0 \
--optimizer 'Adam' \
--mode 'random' \
--lr 0.001 \
--weight_decay 0.00001 \
--schedule_factor 0.1 \
--scheduler_type 'ReduceLROnPlateau' \
--loss_type 'BCEWithLogits' \
--num_workers 8 \
--batch_size 64 \
--eval_batch_size 64 \
--class_type 'multilabel' \
--device_ids 0 1 2 3 \
--momentum 0.9 \
--weak_classes '' \
--chexpert_dataset_type 'pos' \
--log_every_epoch 0 \
--debug 'False' \
--xray_views 'AP' 'PA' \
--query_set_type 'numpy' \
--pick_labels -1 \
--retrain_model_type 'load' \
--test_models_type 'retrained' \
>>train_private_${DATASET}_${timestamp}.txt 2>&1 &
echo train_private_${DATASET}_${timestamp}.txt
```

### The main implementation parts and parameters

The Binary PATE per label and Powerset can be found in
file: `analysis/rdp_cumulative.py`. The functions are `analyze_multilabel()` for
Binary and `analyze_multilabel_powerset` for Powerset.

To activate Binary version use parameter `--class_type 'multilabel'`, for
Powerset `--class_type 'multilabel_powerset'`.

The t-PATE per label can be found in file: `analysis/rdp_cumulative.py` and
function `analyze_tau_pate()`. To activate this version use
parameter `--class_type 'multilabel_tau_pate'`.

To activate the tau-clipping mechanism set the `--class_type`
to `multilabel_tau_data_independent`. Then, `--private_tau_norm 2` (2 is for the
L2 norm) and, e.g., for Pascal VOC the threshold tau is `1.8`. 

### Organization of the Repository

The `analysis` folder contains various files used for the implementation of various differential privacy primitives such as for computing the privacy cost. 
The `architectures` folder contains different model architectures used in the experiments. 
The `dpsgd` folder contains tests for DPSGD which were used to compare the performance of our method and DPSGD. 
The `models` folder contains several helper functions and files for the ensemble based setup in our multi-label PATE implementation. 
