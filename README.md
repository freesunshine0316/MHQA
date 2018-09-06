# MHQA
Multihop QA project, evaluating on Wikihop and WebComplexQA datasets.
Wikihop is taken as the primary dataset for choosing hyperparameters.

We plan to use our [Graph Recurrent Network (GRN)](https://github.com/freesunshine0316/neural-graph-to-seq-mp)  for multi-passage reasoning. The paper title is "[A Graph-to-Sequence Model for AMR-to-Text Generation](https://arxiv.org/abs/1805.02473)", and has been acccepted by ACL18.

## [Wikihop](http://qangaroo.cs.ucl.ac.uk/)

The best published work ([Dhingra et al., NAACL 2018](https://arxiv.org/abs/1804.05922)) on this dataset shows a Dev accuracy of 56%.

A few instances in Wikihop can have more than 100K tokens for their concatenated snippets. One way to handle is to chunk very long snippets:

```
If concatenated size > thres:
    thres_each = thres/(number of snippet)
    for each snippet:
        snippet = snippet[:thres_each]
```

### Onboarding and relavant paper reading (week 0, June 11 -- 13)

### Model development (week 1, June 14 -- 20 and June 21 -- 24, using 4 more days)

### Initial tuning on threshold and hyperparameters (week 2, June 25 -- 27)
- Base model (8k tok thres for passage, bch 20, lr 5e-4, l2 1e-8, w/o GRN, additive attention)<br>
Dev accu: 61.4% (v2 60.1% (10 epoches), v2 NoQrepConcat 60.8% (10 epoches))

- Base + 3k thres for passage <br>
Dev accu: 60.4% (cudnn 60.8%)

- Base + 3k thres for passage + bch 50 <br>
Dev accu: 60.2%

- Base + 3k thres for passage + lr 1e-3 <br>
Dev accu: 61.1%

- Base + 3k thres for passage + lr 1e-3 + l2 1e-3 <br>
Dev accu: 49.2%

- Base + 5k thres for passage <br>
Dev accu: 60.1% (10 epoches)

### Try GRN and other output layers (week 3, June 28 -- July 4)

#### Output layers

- Base 3k thres for passage + symmetric-attn </br>
Dev accu: 60.5%

#### GRN (3k thres, bch 10, lr 5e-4, l2 1e-8)

- GRN-v0: No edges for identical mentions, bugs on window-typed edges  <br>
Dev accu: 63.3%

- GRN-v1: All edges merge together <br>
Dev accu: 62.3%

- GRN-v2: Seperate edges into 3 types: match (same mention), coref, window (mentions that are close to each other) <br>
Dev accu: 62.3%

We find out the current model has too many parameters and is suffered from overfitting problem. 
We are investigating ways to making improvement.

### GRN and Better question representation (+ MP-matching) (week 4, July 5 -- July 11)

Multi-perspective matching doesn't help. There are two possible reasons: first, the questions are very short and are not natural questions. Second, the scale of the dataset isn't large enough (MP-matching introduces more parameters).

#### Multi-step integration ([figure](./grn_integration.pdf))

- GRN-v1: 62.2%

- GRN-v2: (under training)

#### No merge of identical mentions

We suspect that the number of candidate occurances plays an important role on the final performance, so we developed a count-based baseline, which shows a devset accuracy of **25%**. On the other hand, the average number of candidates is **19.1**.

- Baseline: 55.3%

- GRN-v1: (under training)

### ACL travel (week 5, July 12 -- July 20)

## [ComplexWebQuestions](https://www.tau-nlp.org/compwebq)

### First investigation on ComplexWebQuestions (week 6, July 23 -- July 25)

Right now, each instance is associated with 300 web snippets, each of which has roughly 60 tokens. 
This results in around 18K tokens in total for each instance.
So we first need to properly prune the snippets without significantly reducing the coverage.
We tried several ways and show the resulting coverage on the devset (3519 instances, answers are raw without tokenization):

| Method | NER+mention | Mention | Tokenized text | Raw text |
|--------|-------------|---------|----------------|----------|
| Top 10 | 0.6519 | 0.6010 | 0.7545 | 0.7633 |
| Top 20 | 0.7229, 9902 (0.7110, 8943) |
| Top 25 | 0.7391, 12288 (0.7280, 11049) |
| Top 30 | 0.7508, 14962 (0.7403, 13459) | 0.7036 | 0.8238 | 0.8343 |
| Only one query | -- | 0.4564 | 0.7298 | 0.7377 |
| Full (top 100) | 0.8139 | 0.7903 | 0.8770 | 0.8878 |

### Performances on ComplexWebQuestions (weeks 7 and 8, July 26 -- August 8)

#### Wikihop

#### WebComplexQA
Recall is the key factor. By keeping the top 30 snippets, the recall is 73.8% and the highest accuracy now is 33.2% (was 31.8%, baseline is 30.6%).

Formal evaluation on test set.

#### WikiHop
Currently, the best observed devset accuracy is 63.0% (was 62.3%)

System submission to Codalab.

| Method for choosing from filtered| Dev accuracy|
|----------------------------------|-------------|
| First in candidate list | 0.6239 |
| Max occur | 0.6221 |
| Min Occur | 0.6286 |
