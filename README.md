**Beforehand**, we have **2 important database**:

**1、Company database**: We have a huge enterprise database which contains more than 200 million standardized companies from external industrial and commercial channels. This database also describes the relationships between enterprises such as : “parent-subsidiary”, “invest-by”, “share-investor”, “share-manager”, “share-legal-person”.

**2、News database**: We have a news database with large amounts of news that may be related to the enterprise from thousands of websites every day. After a series of cleaning and processing, named entity recognition (NER) methods are then be applied to extract the company entities from the news. One can directly using some open-source NER tools or finetune the model on your own corpus (such as https://github.com/Determined22/zh-NER-TF or https://github.com/baidu/lac). After we extract the company entities, we further do some disambiguation using rule-based methods combined with our Company database. Then each news is mapped to one or more standardized companies.

For research purposes, we only open a typical part of the dataset processed from the above two huge databases. The provided database contains several latest financial news accompanied by corresponding enterprises. News node is linked to company node if the company is detected from the news using the NER tools, and enterprises connected with other enterprises if there is an edge between them from the company database. Some detail content are updated from the paper.

link: https://pan.baidu.com/s/15eJ9hhH0uP2acj2pHRC23A code: v9qc

**Original Experiment File:**

**Edge file:**     data/effect_news_rela_filter_dl.csv   .

**Node property file:**    data/ndatas/       .

Then the while graph infomation are processed into a DGL format file using the opensource tool https://github.com/dmlc/dgl: data/dglgraphs/merge_graph_sigir
 
You can easily load and check this graph using the following code: 
 
glist,_ = dgl.data.utils.load_graphs("data/dglgraphs/merge_graph_sigir") 

print("g.number_of_nodes():", glist[0].number_of_nodes()) 

print("g.number_of_edges():", glist[0].number_of_edges())
`



**Demo code**: `python run_main.py` which includes the key text and graph joint learning part.