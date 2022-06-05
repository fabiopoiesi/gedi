import gdown
import os
import pathlib


def download_unzip(data_dir, dest_dir, url):
    fname = '%s.zip' % dest_dir
    if not os.path.isdir(os.path.join(data_dir, dest_dir)):
        output = os.path.join(data_dir, fname)
        gdown.download(url, output, fuzzy=True)
        cmd = 'unzip %s -d %s' % (output, data_dir)
        os.system(cmd)
        cmd = 'rm %s' % output
        os.system(cmd)
    else:
        print('[i] directory <%s> already existing in <data> (<%s> will not be downloaded)' % (dest_dir, fname))


curr_dir = pathlib.Path(__file__).parent.resolve()
data_dir = os.path.join(curr_dir, 'data')
os.makedirs(data_dir, exist_ok=True)


'''
pretrained models
'''

download_unzip(data_dir=data_dir,
               dest_dir='chkpts',
               url='https://drive.google.com/file/d/1Lpep5QigALjk60h8bNJAUH3DnxtnGcZX/view?usp=sharing')


'''
assets
'''
download_unzip(data_dir=data_dir,
               dest_dir='assets',
               url='https://drive.google.com/file/d/1EEMzusKM2QPi9u3o3qsgAwPPX6c_SlBd/view?usp=sharing')



