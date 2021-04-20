import numpy as np 
import h5py, subprocess

# Generates testing datasets from Xia, Heming, et al doi:10.1103/PhysRevD.103.024040.
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generates testing datasets')
    parser.add_argument('-n',default='1', type=int, help='Generates the nth testing dataset from the paper')
    args=parser.parse_args()
    if args.n==1:
        snr_testing_1=np.array([7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15])
        for snr in snr_testing_1:
            with open('config_files/params_1_testing.ini','r') as f:
                params_1_testing_ini=f.readlines()
            params_1_testing_ini[16-1] = 'injection_snr = {}\n'.format(snr)
            params_1_testing_ini[93-1] = 'min-injection_snr = {}\n'.format(snr)
            params_1_testing_ini[94-1] = 'max-injection_snr = {}\n'.format(snr+0.0001)

            with open('config_files/params_1_testing.json','r') as f:
                params_1_testing_json=f.readlines()
            params_1_testing_json[11-1] = "\t\"output_file_name\": \"data_1_testing_snr={}.hdf\"\n".format(snr)
            
            with open('config_files/params_1_testing.ini','w') as f:
                f.writelines(params_1_testing_ini)
            with open('config_files/params_1_testing.json','w') as f:
                f.writelines(params_1_testing_json)
            
            subprocess.run(["python2","generate_sample.py","--config-file=params_1_testing.json"])
    elif args.n==2:
        mass_testing_2=np.array(np.arange(10,80,10))
        for i in range(mass_testing_2.size):
            for j in range(i+1):
                with open('config_files/params_2_testing.ini','r') as f:
                    params_2_testing_ini=f.readlines()

                params_2_testing_ini[7-1] = 'mass1 = {}\n'.format(mass_testing_2[i])
                params_2_testing_ini[67-1] = 'min-mass1 = {}\n'.format(mass_testing_2[i])
                params_2_testing_ini[68-1] = 'max-mass1 = {}\n'.format(mass_testing_2[i]+0.0001)

                params_2_testing_ini[8-1] = 'mass2 = {}\n'.format(mass_testing_2[j])
                params_2_testing_ini[74-1] = 'min-mass2 = {}\n'.format(mass_testing_2[j])
                params_2_testing_ini[75-1] = 'max-mass2 = {}\n'.format(mass_testing_2[j]+0.0001)

                with open('config_files/params_2_testing.json','r') as f:
                    params_2_testing_json=f.readlines()
                params_2_testing_json[11-1] = "\t\"output_file_name\": \"data_2_testing_m1={0}_m2={1}.hdf\"\n".format(mass_testing_2[i],mass_testing_2[j])
                params_2_testing_json[8-1] = "\t\"n_injection_samples\": {},\n".format(278)
                params_2_testing_json[9-1] = "\t\"n_noise_samples\": {},\n".format(278)

                with open('config_files/params_2_testing.ini','w') as f:
                    f.writelines(params_2_testing_ini)
                with open('config_files/params_2_testing.json','w') as f:
                    f.writelines(params_2_testing_json)
                subprocess.run(["python2","generate_sample.py","--config-file=params_2_testing.json"])
    elif args.n==3:
        spin_testing_3=np.array(np.linspace(-0.998,0.998,8))
        for i in range(8):
            for j in range(8):
                with open('config_files/params_3_testing.ini','r') as f:
                    params_3_testing_ini=f.readlines()

                params_3_testing_ini[9-1] = 'spin1z = {}\n'.format(spin_testing_3[i])
                params_3_testing_ini[81-1] = 'min-spin1z = {}\n'.format(spin_testing_3[i])
                params_3_testing_ini[82-1] = 'max-spin1z = {}\n'.format(spin_testing_3[i]+0.0001)

                params_3_testing_ini[10-1] = 'spin2z = {}\n'.format(spin_testing_3[j])
                params_3_testing_ini[88-1] = 'min-spin2z = {}\n'.format(spin_testing_3[j])
                params_3_testing_ini[89-1] = 'max-spin2z = {}\n'.format(spin_testing_3[j]+0.0001)

                with open('config_files/params_3_testing.json','r') as f:
                    params_3_testing_json=f.readlines()
                params_3_testing_json[11-1] = "\t\"output_file_name\": \"data_3_testing_s1={0}_s2={1}.hdf\"\n".format(spin_testing_3[i],spin_testing_3[j])
                params_3_testing_json[8-1] = "\t\"n_injection_samples\": {},\n".format(157)
                params_3_testing_json[9-1] = "\t\"n_noise_samples\": {},\n".format(157)
                with open('config_files/params_3_testing.ini','w') as f:
                    f.writelines(params_3_testing_ini)
                with open('config_files/params_3_testing.json','w') as f:
                    f.writelines(params_3_testing_json)
                subprocess.run(["python2","generate_sample.py","--config-file=params_3_testing.json"])
    elif args.n==4:
        subprocess.run(["python2","generate_sample.py","--config-file=params_4_testing.json"])
    elif args.n==5:
        snr_testing_5=np.array([7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15])
        for snr in snr_testing_5:
            with open('config_files/params_5_testing.ini','r') as f:
                params_5_testing_ini=f.readlines()
            params_5_testing_ini[16-1] = 'injection_snr = {}\n'.format(snr)
            params_5_testing_ini[94-1] = 'min-injection_snr = {}\n'.format(snr)
            params_5_testing_ini[95-1] = 'max-injection_snr = {}\n'.format(snr+0.0001)

            with open('config_files/params_5_testing.json','r') as f:
                params_5_testing_json=f.readlines()
            params_5_testing_json[11-1] = "\t\"output_file_name\": \"data_5_testing_snr={}.hdf\"\n".format(snr)
            
            with open('config_files/params_5_testing.ini','w') as f:
                f.writelines(params_5_testing_ini)
            with open('config_files/params_5_testing.json','w') as f:
                f.writelines(params_5_testing_json)
            
            subprocess.run(["python2","generate_sample.py","--config-file=params_5_testing.json"])