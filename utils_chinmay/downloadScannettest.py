import subprocess
count = 0
with open('/home/student/Mask3D/data/raw/scannet/ScanNet/Tasks/Benchmark/scannetv2_train.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        if count < 15:
            count += 1
            continue
        elif count < 50:
            count += 1
        else:
            exit()
        cmd = 'printf \'y\\nn\\n\' | python3 /home/student/downloadScannet.py -o /home/student/Scannet2 --id ' + str(line.strip())
        print(cmd, '  ', count)
        proc = subprocess.Popen(cmd, shell=True, stdin = subprocess.PIPE, stdout= subprocess.PIPE)
        proc.communicate()
     
        
        
