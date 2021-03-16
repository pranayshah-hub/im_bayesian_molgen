### **Implementation instructions for experiments of the IM-B2Q  set of models. Code written in Python 3.7.9.**

#### **Computing infrastructure:**
Experiments were run on a GPU host of 24-core/48 thread Intel Xeon CPUs with 256GB RAM. The graphics cards used were the Nvidia Tesla T4 Tensor Core GPU
or the Nvidia TITAN Xp Graphics Card, with a CUDA toolkit version 10.1.

#### **Requirements:**
All packages and dependencies are specified in requirements.txt. To create the environment, run either:

*conda create --name <env_name> --file requirements.txt*

or

*pip install -r requirements.txt*

#### **Known issues:**
The following exception is known to occur due to incomplete support in the RDKit package. It occurs rarely when a chemical operation or state is not recognised or supported.

'Pre-condition Violation:
bad bond type.
Violation occurred on line ---- in file ./GraphMol/Canon.cpp.
Failed Expression: bond$\rightarrow$getBondType()==Bond::SINGLE||bond$\rightarrow$getIsAromatic()'

#### **Results:**
| Method            |   | Top-3 Max         |   | Avg   |
|-------------------|---|-------------------|---|-------|
| IM-B2Q            |   | 0.948 0.948 0.948 |   | 0.903 |
| B2Q               |   | 0.946 0.945 0.945 |   | 0.801 |
| BBQ               |   | 0.943 0.941 0.939 |   | 0.810 |
| BBQ+LR            |   | 0.941 0.940 0.938 |   | 0.853 |
| BBQ+PS            |   | 0.776 0.759 0.705 |   | 0.372 |
| B2Q Avg-Sample    |   | 0.943 0.943 0.943 |   | 0.846 |
| B2Q Sample-Sample |   | 0.939 0.930 0.923 |   | 0.811 |
| B2Q Avg-Avg       |   | 0.938 0.926 0.921 |   | 0.767 |
| IDS-e1            |   | 0.948 0.946 0.944 |   | 0.793 |
| IDS-e2            |   | 0.941 0.941 0.941 |   | 0.744 |
| MolDQN IDS-e1     |   | 0.942 0.936 0.934 |   | 0.845 |
| IM InfoGain-e1    |   | 0.929 0.929 0.929 |   | 0.825 |
| IM MolDQN IDS-e1  |   | 0.928 0.928 0.928 |   | 0.840 |
| IM IDS-e2         |   | 0.878 0.868 0.840 |   | 0.697 |
| IM e1             |   | 0.856 0.849 0.849 |   | 0.832 |
| IM IDS-e1 + LR    |   | 0.801 0.800 0.791 |   | 0.703 |

#### **Commands to reproduce results:**
IM-B2Q: \
python optimize_qed.py --model_dir=${OUTPUT_DIR} --hparams="./configs/im_ids_epi1.json" --model_type="bdqn_ids"

B2Q: \
python optimize_qed.py --model_dir=${OUTPUT_DIR} --hparams="./configs/sample_avg.json" --model_type="bdqn"

BBQ: \
python optimize_qed.py --model_dir=${OUTPUT_DIR} --hparams="./configs/molbbq.json" --model_type="bdqn"

BBQ+LR: \
python optimize_qed.py --model_dir=${OUTPUT_DIR} --hparams="./configs/molbbq_lr.json" --model_type="bdqn"

BBQ+PS: \
python optimize_qed.py --model_dir=${OUTPUT_DIR} --hparams="./configs/molbbq_ps.json" --model_type="bdqn_ps"

B2Q Avg-Sample: \
python optimize_qed.py --model_dir=${OUTPUT_DIR} --hparams="./configs/ avg_sample.json" --model_type="bdqn"

B2Q Sample-Sample: \
python optimize_qed.py --model_dir=${OUTPUT_DIR} --hparams="./configs/sample_sample.json" --model_type="bdqn"

B2Q Avg-Avg: \
python optimize_qed.py --model_dir=${OUTPUT_DIR} --hparams="./configs/avg_avg.json" --model_type="bdqn"

IDS-e1: \
python optimize_qed.py --model_dir=${OUTPUT_DIR} --hparams="./configs/ids_epi1.json" --model_type="bdqn_ids"

IDS-e1: \
python optimize_qed.py --model_dir=${OUTPUT_DIR} --hparams="./configs/ids_epi2.json" --model_type="bdqn_ids"

MolDQN IDS-e1: \
python optimize_qed.py --model_dir=${OUTPUT_DIR} --hparams="./configs/ids_moldqn_epi1.json" --model_type="dqn_ids"

IM InfoGain-e1: \
python optimize_qed.py --model_dir=${OUTPUT_DIR} --hparams="./configs/im_infogain_epi1.json" --model_type="bdqn_ids"

IM MolDQN IDS-e1: \
python optimize_qed.py --model_dir=${OUTPUT_DIR} --hparams="./configs/im_moldqn_ids_epi1.json" --model_type="dqn_ids"

IM IDS-e2: \
python optimize_qed.py --model_dir=${OUTPUT_DIR} --hparams="./configs/im_ids_epi2.json" --model_type="bdqn_ids"

IM e1: \
python optimize_qed.py --model_dir=${OUTPUT_DIR} --hparams="./configs/im_epi1.json" --model_type="bdqn_ids"

IM IDS-e1 + LR: \
python optimize_qed.py --model_dir=${OUTPUT_DIR} --hparams="./configs/im_ids_epi1_lr.json" --model_type="bdqn_ids"


