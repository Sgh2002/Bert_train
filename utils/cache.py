from utils.oss_utils import OssUtils
from pathlib import Path
import os
import torch.distributed as dist

class Cache(object):

    def __init__(self, ak,sk,bucket, endpoint,cache_dir=None) -> None:
        if cache_dir is None:
            self.cache_dir = Path(os.path.join(os.path.expanduser("~"), ".cache",'qtc'))
            print(f'default cache dir : {self.cache_dir}')

        else:
            self.cache_dir=Path(cache_dir)
        self.oss = OssUtils(ak,sk,bucket, endpoint)
    
    def get_cached_file_path(self, file_path):

        if file_path == None:
            return file_path

        if file_path[0] in ('/','.'):
            return file_path


        file_path = Path(file_path)
        the_file = self.cache_dir/file_path.stem
        if the_file.exists():
            print(f'find cached file {the_file}')
        else:

            rank = os.getenv('RANK', 'null')
            local_rank = os.getenv('LOCAL_RANK', 'null')
           
            if (dist.is_initialized() and local_rank=='0') or dist.is_initialized()==False:

                #download
                print(f'rank {rank}, local rank {local_rank}:start to download files {file_path}')

                the_file.mkdir(parents=True,exist_ok=True)
                self.oss.download_folder(file_path, the_file)
        
        if dist.is_initialized():
            dist.barrier()
        
        return the_file


        
