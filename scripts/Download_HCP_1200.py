from subprocess import  PIPE, run
import numpy as np
import os, argparse

from multiprocessing import Pool

parser = argparse.ArgumentParser(description='Download HCP 1200 fMRI data through AWS.')
parser.add_argument("-n", "--node",type=str,
                     help="insert the number of nodes in SKKU HPC server. (recomanded among 14 to 18)")
parser.add_argument("-w", "--workers", type=int,
                     help="insert number of threads to use.")
args = parser.parse_args()

def download_subj_fmri(subj_batch, save_path):
    for subj_num in subj_batch:
        path = save_path+subj_num

        if not os.path.isdir(path) : os.mkdir(path)

        command = 'aws s3 sync s3://hcp-openaccess/HCP_1200/{}/ {}'.format(subj_num,path).split(' ')
        _ = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)

def download_HCP_1200(HPC_node_num, workers, save_path='/cocoanlab2/GPU1_sync/data/HCP_1200/'):
    save_path += '/' if save_path[-1] != '/' else ''
    
    HPC_node = {str(node_num) : idx for idx, node_num in enumerate(range(14,19))}
    
    subj_list = open(save_path+'subj_num.txt').readlines()
    subj_list = sorted([v.replace(' ','').replace('\n','') for v in subj_list])
    subj_list = np.array(subj_list)
    subj_batchset = np.array(np.array_split(subj_list, len(HPC_node)*workers)).reshape(len(HPC_node),workers)

    multiprocessor = Pool(processes = workers)
    
    for batch in subj_batchset[HPC_node[HPC_node_num]]: 
        multiprocessor.apply_async(download_subj_fmri, args=(batch,save_path,))
    multiprocessor.close()
    multiprocessor.join()
    
if __name__ == "__main__":
    download_HCP_1200(HPC_node_num=args.node, workers=args.workers)