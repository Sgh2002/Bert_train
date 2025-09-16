import oss2
from pathlib import Path

class OssUtils(object):

    def __init__(self, ak,sk, bucket, endpoint) -> None:
        self.ak=ak
        self.sk=sk
        self.bucket_str = bucket
        self.endpoint=endpoint
        self.auth = oss2.Auth(ak, sk)
        self.bucket = oss2.Bucket(self.auth, endpoint, bucket)

    def list_objects(self, prefix):

        objects = []
        #遍历文件
        for obj in oss2.ObjectIterator(self.bucket, prefix = prefix,max_keys=100): 
            
            if obj.is_prefix() or obj.key.endswith('/') :  # 判断obj为文件夹。
                pass
            else:                # 判断obj为文件。
                objects.append(obj.key)
        return objects
    
    def open_oss_file(self, file_name,mode='r',selected_col=None):
        
        file_type = file_name.split('.')[-1]
        
        if mode=='r':
            if file_type in ('json','txt'):
                fp = OssJsonFileReader(file_name,self.bucket,selected_col)
            elif file_type == 'csv':
                fp = OssCsvFileReader(file_name, self.bucket,selected_col)
            else:
                raise ValueError('file ext must named with one of json ,txt, csv')

        elif mode == 'w':
            fp = OssFileWriter(file_name,self.bucket)

        return fp
    
    def download_folder(self, oss_path,local_path):

        for obj in oss2.ObjectIteratorV2(self.bucket, prefix=oss_path):
            # obj.key是文件名BERT-12L-12H-768-Query_Title_128_2_torch/
            # 下载文件到本地
            if obj.is_prefix() or obj.size==0:
                continue
            local_file = Path(local_path)/(obj.key.split('/')[-1])
            self.bucket.get_object_to_file(obj.key, local_file)


    def download(self, oss_file, local_file):
        self.bucket.get_object_to_file(oss_file, local_file)        


class OssFileWriter(object):

    def __init__(self, file_name,bucket) -> None:
        self.bucket = bucket
        self.file_name = file_name 
        self.result=None
        bucket.delete_object(file_name)
    
    def write(self, content):

        
        if self.result==None:
            pos = 0
        else:
            pos = self.result.next_position
        
        self.result = self.bucket.append_object(self.file_name, pos, content)  


class OssCsvFileReader(object):
    def __init__(self, file_name, bucket,selected_col) -> None:
        
        self.bucket = bucket
        self.csv_meta_params = {'RecordDelimiter': '\n','OverwriteIfExists':'False'}
        self.csv_header = bucket.create_select_object_meta(file_name,self.csv_meta_params)
        self.file_name=file_name
        self.selected_col = selected_col
    
    def __iter__(self):
        splits = self.csv_header.splits

        for slice in range(0, splits):
            print(slice)

            select_csv_params = {'CsvHeaderInfo': 'None',
                    'RecordDelimiter': '\n',
                    'SplitRange':(slice, slice)}
            if self.selected_col is not None:
            
                cols = []
                for col in self.selected_col:
                    cols.append(f's.{col}') 
                
                sel = ','.join(cols)
            
            else:
                sel = "*"
                
            sql = f'select {sel} from ossobject s'
            

            object = self.bucket.select_object(self.file_name,sql=sql,select_params=select_csv_params)
            content  = object.read().decode('utf-8').split('\n')
            for line in content:
                yield line 
       
class OssJsonFileReader(object):

    def __init__(self, file_name, bucket,selected_col) -> None:
        
        self.bucket = bucket
        select_json_params = {'Json_Type': 'LINES','OverwriteIfExists':'False'}
        self.json_header = bucket.create_select_object_meta(file_name,select_json_params)
        self.file_name=file_name
        self.selected_col = selected_col
        
    
    def __iter__(self):
        splits = self.json_header.splits

        for slice in range(0, splits):
            select_json_params = {'Json_Type': 'LINES','SplitRange':(slice, slice)}

            if self.selected_col is not None:
            
                cols = []
                for col in self.selected_col:
                    cols.append(f's.{col}') 
                
                sel = ','.join(cols)
            
            else:
                sel = "*"
                
            sql = f'select {sel} from ossobject s'
            

            object = self.bucket.select_object(self.file_name,sql=sql,select_params=select_json_params)
            content  = object.read().decode('utf-8').split('\n')
            for line in content:
                yield line 
    

        



if __name__ == "__main__":
    # oss= 
    objects = oss.list_objects('QTC/yongbo/train_idx_logits_new0301_part_small_body.json')
    print(objects)

    selected_col = ['query','title']

    for f in objects:
        oss_file = oss.open_oss_json_file(f,selected_col)

        for line in oss_file:
            print(line)

 