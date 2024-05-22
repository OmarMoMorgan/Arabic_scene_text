from . import pipeline1

def build_transforms(pipeline_name,config): 
    if pipeline_name == 'pipeline1':
        return pipeline1.get_transforms(**config) # returns train and test transforms