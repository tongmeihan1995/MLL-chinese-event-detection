Chinese Event Detection Using Lattice LSTM based Multi-task Learning (MLL)
====

Models and results can be found at our KSEM 2020 paper Improving Low-Resource Chinese Event Detection with Multi-task Learning. 

Requirement:
======
	Python: 2.7   
	PyTorch: 0.3.0 

Input format:
======
BMSE tag scheme, with each character its label for one line. Sentences are splited with a null line.

	他 0
	说 0
	， 0
	原 0
	本 0
	活 0
	泼 0
	好 0
	动 0
	的 0
	他 0
	如 0
	今 0
	半 B-Injure
	身 M-Injure
	不 M-Injure
	遂 E-Injure
	. 0


Pretrained Embeddings:
====
The pretrained character and word embeddings are the same with the embeddings in the baseline of [RichWordSegmentor](https://github.com/jiesutd/RichWordSegmentor)

Character embeddings: [gigaword_chn.all.a2b.uni.ite50.vec](https://pan.baidu.com/s/1pLO6T9D)

Word(Lattice) embeddings: [ctb.50d.vec](https://pan.baidu.com/s/1pLO6T9D)

Multi-learning Tasks:
====
For NER task, we use the MSRA corpus

For Mask Word Prediction task, we use the lastest Wiki corpus

How to run the code?
====
1. Download the character embeddings and word embeddings and put them in the `data` folder.
2. Download the NER corpus and Wiki corpus and put them in the `data` folder.
3. For training, run the script `run_main.sh`
4. For testing, run the script `run_test.sh`


Cite: 
========
Please cite our KSEM 2020 paper:

    @article{tong2020improving,  
     title={Improving Low-Resource Chinese Event Detection with Multi-task Learning},  
     author={Meihan Tong, Bin Xu, Shuai Wang, Hou Lei, Juaizi Li},  
     booktitle={Knowledge Science, Engineering and Management(KSEM))},
     year={2020}  
    }
