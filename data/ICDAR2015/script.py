import io
import glob

file_list = glob.glob('*.txt')

for filename in file_list:
    fid = io.open(filename, 'r', encoding='utf-8-sig')
    content = fid.readlines()
    fid.close()
    newname = '../Annotations/%06d.gt'%int(filename[7:-4])
    file = open(newname, 'w')
    for line in content:
        a = line.split(',')
        string = '0 '
        if '###' in a[8]:
            string = '1 '
        for i in range(8):
            string += a[i] + ' '
        file.write(string + '\n')
    file.close()
        
