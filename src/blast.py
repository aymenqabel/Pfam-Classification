import pandas as pd

from Bio.Blast.Applications import NcbiblastpCommandline

from src.utils import compute_metrics

def run_blast(path_models, subject,file_test, test='test', evalue = 1e-3):
    """This function runs blast to compute sequence alignment between test sequences and train sequences.

    Args:
        path_models (str): _description_
        subject (str): Path to the train fasta file
        file_test (str): Path to the test fasta file
        test (str, optional): _description_. Defaults to 'test'.
        evalue (float, optional): _description_. Defaults to 1e-3.

    Returns:
        str: path to the resulting table
    """
    fwd_out = path_models + f"{test}_{evalue}1111111" # chnage with os path join
    blastp_path = '/usr/local/ncbi/blast/bin/blastp'
    blastp = NcbiblastpCommandline(cmd = blastp_path,
                                   query= file_test, 
                                   subject=subject,
                                   out=fwd_out,
                                    outfmt="6 qseqid sseqid pident qcovs qlen slen length bitscore evalue",
                                    max_target_seqs=1, evalue = evalue,
                                    num_threads = 8)
    _, _ = blastp()
    return fwd_out

def read_results_from_dataframe(fwd_out):
    """This function reads the resulting dataframe of blast

    Args:
        fwd_out (str): Path to the resulting blast dataframe

    Returns:
        Pandas DataFrame: Resulting dataframe
    """
    fwd_results = pd.read_csv(fwd_out, sep="\t", header=None)
    headers = ["query", "subject", "identity", "coverage",
            "qlength", "slength", "alength",
            "bitscore", "E-value"]
    fwd_results.columns = headers
    fwd_results['norm_bitscore'] = fwd_results.bitscore/fwd_results.qlength
    return fwd_results

def get_results(fwd_results, train_dict, test_dict):
    """Compute the results of blast predictions

    Args:
        fwd_results (dataframe): dataframe with blast results
        train_dict (dict): dictionary of train sequence names with their families
        test_dict (dict): dictionary of test sequence names with their families

    Returns:
        _type_: _description_
    """
    predictions = dict.fromkeys(dict(test_dict), 'N')
    for row in fwd_results.values:
        predictions[row[0]] = train_dict[row[1]]
    names = []
    y_true = []
    y_pred = []
    for prot in test_dict.keys():
        y_true.append(test_dict[prot])
        y_pred.append(predictions[prot])
        names.append(prot)
    return y_true, y_pred, names

def load_labels():
    """Load dictionary of sequence_name: family

    Returns:
        Dict: dictionary of the families
    """
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
    train_dict = df_train[['sequence_name', 'family_accession']].set_index('sequence_name').to_dict()['family_accession']
    test_dict = df_test[['sequence_name', 'family_accession']].set_index('sequence_name').to_dict()['family_accession']
    return train_dict, test_dict

if __name__=="__main__":
    train_dict, test_dict = load_labels()
    SUBJECT = "data/train_file.fasta"
    QUERY ="data/test_file.fasta"   
    fwd_out = run_blast('models/', SUBJECT, QUERY, f"test", evalue=1e-3)
    fwd_results = read_results_from_dataframe(fwd_out)
    y_true, y_pred, names = get_results(fwd_results, train_dict, test_dict)
    compute_metrics(y_true, y_pred,path_results='results/',  method = "BLAST", names=names)
    