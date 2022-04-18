# Author: Rehan
# Helper functions

import os
import subprocess
from pathlib import Path


def run_scorer(params):
    perl_script = subprocess.check_output(params)
    #     perl_script = subprocess.check_output()
    perl_script = perl_script.decode("utf-8")
    index = perl_script.find("====== TOTALS =======")
    return perl_script[index:]


def generate_results(gold_key_file, system_key_file):
    """

    Parameters
    ----------
    gold_key_file: str
    system_key_file: str

    Returns
    -------
    None
    """
    scorer = str(Path(__file__).parent) + '/scorer/scorer.pl'
    params = ["perl", scorer, "bcub", gold_key_file, system_key_file]
    print("BCUB SCORE")
    print(run_scorer(params))

    params[2] = "muc"
    print("MUC SCORE")
    print(run_scorer(params))


def generate_key_file(coref_map_tuples, name, out_dir, out_file_path):
    """

    Parameters
    ----------
    coref_map_tuples: list
    name: str
    out_dir: str
    out_file_path: str

    Returns
    -------
    None
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    clus_to_int = {}
    clus_number = 0
    with open(out_file_path, 'w') as of:
        of.write("#begin document (%s);\n" % name)
        for i, map_ in enumerate(coref_map_tuples):
            en_id = map_[0]
            clus_id = map_[1]
            if clus_id in clus_to_int:
                clus_int = clus_to_int[clus_id]
            else:
                clus_to_int[clus_id] = clus_number
                clus_number += 1
                clus_int = clus_to_int[clus_id]
            of.write("%s\t0\t%d\t%s\t(%d)\n" % (name, i, en_id, clus_int))
        of.write("#end document\n")
