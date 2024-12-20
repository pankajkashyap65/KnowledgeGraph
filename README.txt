## README: Entity Selection Framework for Ad-Hoc Retreival

**Purpose:**

This Python script implements an entity acquisition algorithm (ESF) that utilizes a graph clustering framework to discover semantic relationships between entities and identify highly coherent entities relevant to a given query. By leveraging techniques such as K-means clustering, entity annotation, and graph-based clustering, it systematically selects entities from diverse resources, including queries, Wikipedia, Wikidata, and pseudo-relevance feedback documents. The graph representation captures semantic relatedness between entities, where nodes represent entities and edges reflect their coherence, minimizing query intent drift during retrieval. This approach ensures more accurate query expansion and improves the precision of similarity search and ranking.

**Prerequisites:**

- **Python:** Ensure you have Python 3.x installed.
- **Libraries:** Install the following Python libraries using `pip`:
    - `spacy`
    - `sklearn`
    - `matplotlib`
    - `numpy`
    - `statsmodels`
    - `pandas`
    - `networkx`
    - `wikidata`
    - `tagme`
    - `wikimapper`
    - `bs4`
	
- **Indri Search Engine:** Make sure you have Indri installed and configured. Set the paths to `indrirunquery_command` and `dumpindex_command` in the code accordingly.
- **Transe Embeddings:** Download and place the pre-trained Transe entity embeddings (entity2vec.bin and entity2id.pickle) in the specified directory.
Link of the paper: 
	https://papers.nips.cc/paper_files/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html
TransE Link:
	https://everest.hds.utc.fr/doku.php
	
**Datasets & Queries:**

- Prepare the datasets' index file, corresponding queries and corresponding relevant documents. The script supports multiple datasets, including gov2, cw12, trec678, robust, MS-MARCO and cw09.
- Ensure the dataset is in the expected format, with queries in the form "query_number=>query_text".
- Place the query files in the "Topics" directory.

-Important Links: can be obtained from: https://trec.nist.gov/data.html
	*Topics (query) file needed for the datasets:
	CW09 (https://trec.nist.gov/data/webmain.html)
	CW12 (https://trec.nist.gov/data/webmain.html)
	ROBUST (https://trec.nist.gov/data/robust.html)
	GOV2 (https://trec.nist.gov/data/terabyte.html) 
	*Datasets CW09, CW12, ROBUST, & GOV2 can be obtained from: https://trec.nist.gov/
	*For MS-MARCO dataset and queries: https://microsoft.github.io/msmarco/
	
**Execution:**

1. **Configure Paths:**
   - Set the paths to `indrirunquery_command`, `dumpindex_command`, and the Transe embedding files in the code (as per your directories).
   - Adjust the dataset-specific paths and filenames as needed.

2. **Run the Script:**
   - Open a terminal or command prompt.
   - Navigate to the directory containing the script.
   - Execute the script using the following command:

     ```bash
     python script_name.py
     ```
     Replace `script_name.py` with the actual name of your Python script (ESF_NDM or ESF_WIA).

3. **Specify Dataset:**
   - The script will prompt you to select a dataset. Choose the appropriate number corresponding to your dataset (1 for gov2, 2 for cw12, etc.).
   - You can also choose to run the code one dataset at a time from the main.

4. **View Output:**
   - The script will process the queries in your dataset, perform entity clustering and similarity search, and generate output files containing the reformulated queries.

**Output:**

- The script will create output files in the "query_G/prob_dist_ESF_Softmax" directory.
- The files will contain the reformulated queries in Indri format.

*Results files dataset-wise are uploaded in "ExperimentResults" folder.
- How to Read:
	-The result file contains 6 columns (C1-C6)
	-C1- Query No.
	-C2- Query internal representation no. assigned by INDRI (INDRI uses for its internal processing purpose).
	-C3- Document IDs retrieved for a specific query no. (C1).
	-C4- Document Ranking (based on relevance).
	-C5- INDRI relevance score of a document (C3) with query (C1).
	-C6- RunName can be customzied (renamed) based on datasets for better understanding, or can be kept default "runName".

-How to Run: 
	- use trec.eval (https://trec.nist.gov/trec_eval/index.html) and gdeval.pl (https://trec.nist.gov/data/test_coll.html) to compare the retrieved
	  documents with the gold standard list (datawise)
	- Example: The below run command compares the retrieved document result file (here for the dataset ROBUST) with the gold standard list (qrel.all) and calculates/writes
		the result in a new MAP file.
		~/indri-5.12/trec_eval.9.0/gdeval.pl ~/indri-5.12/trec_eval.9.0/qrel.all /home/pankaj/indri-5.12/query/ESF/robust/robust_rq > /home/pankaj/indri-5.12/query/ESF/robust/robust_MAP
	- qrel.all can be obtained from (https://trec.nist.gov/data.html):
		CW09 (https://trec.nist.gov/data/webmain.html)
		CW12 (https://trec.nist.gov/data/webmain.html)
		ROBUST (https://trec.nist.gov/data/robust.html)
		GOV2 (https://trec.nist.gov/data/terabyte.html) 
	These individual links contain the gold standard results for respective query.
	
**Additional Notes:**

- The script assumes that the Indri index files are located in the specified directories.
- For more details on the code's functionality, refer to the comments within the script.




