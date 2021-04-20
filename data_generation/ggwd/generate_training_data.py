import subprocess
# Generates training datasets from Xia, Heming, et al doi:10.1103/PhysRevD.103.024040.
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Generates testing datasets')
    parser.add_argument('-n',default='1', type=int, help='Generates the nth testing dataset from the paper')
    args=parser.parse_args()
    
    if n==1:
        subprocess.run(["python2","generate_sample.py","--config-file=params_1.json"])
    elif n==2:
        subprocess.run(["python2","generate_sample.py","--config-file=params_2.json"])
    elif n==3:
        subprocess.run(["python2","generate_sample.py","--config-file=params_3.json"])
    elif n==4:
        subprocess.run(["python2","generate_sample.py","--config-file=params_4.json"])
    elif n==5:
        subprocess.run(["python2","generate_sample.py","--config-file=params_5.json"])
