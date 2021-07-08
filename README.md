# Sparse and Structured Visual Attention
Implementation of the experiments for visual question answering with sparse and structured visual attention.


## Visual Question Answering

### Requirements

We recommend to follow the procedure in the official [MCAN](https://github.com/MILVLG/mcan-vqa) repository in what concerns software and hardware requirements. We also use the same setup - see there how to organize the `datasets` folders. The only difference is that we also use grid features; you can download them from [here](https://github.com/facebookresearch/grid-feats-vqa).

Run
```entmax
pip install entmax
```
to install the entmax package.

### Training

To train the models in the paper, run this command:

```train
python3 run.py --RUN=train --M='mca' --gen_func=<ATTENTION> --SPLIT=train --features=<FEATURES>
```
with ```<ATTENTION>={'softmax', 'sparsemax', 'tvmax'}``` to train the model with softmax, sparsemax, or TVmax attention, and ```<FEATURES>={'grid', 'bounding_boxes'}``` to train the model with grid features or bounding box features.



### Evaluation

The evaluations of both the VQA 2.0 *test-dev* and *test-std* splits are run as follows:

```eval
python3 run.py --RUN=test --CKPT_V=<VERSION> --CKPT_E=<EPOCH TO LOAD> --M='mca' --gen_func=<ATTENTION> --features=<FEATURES>

```
and the result file is stored in ```results/result_test/result_run_<'PATH+random number' or 'VERSION+EPOCH'>.json```. The obtained result json file can be uploaded to [Eval AI](https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview) to evaluate the scores on *test-dev* and *test-std* splits.

# Citation

    @inproceedings{martins2021sparse,
      author    = {Martins, Pedro Henrique and Niculae, Vlad and Marinho, Zita and  Martins, Andr{\'e} FT},
      title     = {Sparse and Structured Visual Attention},
      booktitle = {Proc. ICIP},
      year      = {2021}
    }
