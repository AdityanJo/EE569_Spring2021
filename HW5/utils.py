# Adityan Jothi
# USC ID 8162222801
# jothi@usc.edu

configuration={}

configuration['activation']='relu'
configuration['number_of_classes']=10
configuration['dataset_type'] = 'MNIST'
configuration['initialization_method']='xavier_normal'
configuration['rate_learn'] = 1e-3
configuration['rate_decay'] = 0
configuration['batch_size'] = 256
configuration['augmentation'] = None # choose between none and 'negative' 

configuration['model_path']='mnist_1ctrain'
configuration['input_size']=(28,28)
configuration['num_workers']=1
configuration['max_epoch_val']=50
configuration['enable_log']=True
configuration['use_gpu']=True
configuration['weight_decay']=0
