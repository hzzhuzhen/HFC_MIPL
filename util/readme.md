1. **Stage 1:** run drama_stage_1.m（such as ```drama_stage_1('emotion_4')```
    1. A *.mat data file will be read, which contains three matrices X, Y, Y_P (partial target).
    2. The results will be output to the same place as input, with file name *_conf.mat. It includes the confidence matrix and X, Y.
2. **Stage 2:** Run main.py with *_conf.mat.



We recommend to fine-tune the depth of decision tree (DecisionTreeRegressor.max_depth). Depth of 3~8 is a good choice in practice.

This code package can be used freely for academic, non-profit purposes. The related paper is as follows,

```
@inproceedings{DBLP:conf/ijcai/Wang0ZZHC19,
  author    = {Haobo Wang and
               Weiwei Liu and
               Yang Zhao and
               Chen Zhang and
               Tianlei Hu and
               Gang Chen},
  editor    = {Sarit Kraus},
  title     = {Discriminative and Correlative Partial Multi-Label Learning},
  booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on
               Artificial Intelligence, {IJCAI} 2019, Macao, China, August 10-16,
               2019},
  pages     = {3691--3697},
  publisher = {ijcai.org},
  year      = {2019},
  url       = {https://doi.org/10.24963/ijcai.2019/512},
  doi       = {10.24963/ijcai.2019/512},
  timestamp = {Sun, 25 Oct 2020 23:13:55 +0100},
  biburl    = {https://dblp.org/rec/conf/ijcai/Wang0ZZHC19.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```