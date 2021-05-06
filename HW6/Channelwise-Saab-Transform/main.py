import numpy as np
from tensorflow.keras.datasets import mnist,fashion_mnist
from skimage.util import view_as_windows
from pixelhop import Pixelhop
from skimage.measure import block_reduce
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
import warnings, gc
import time
import pickle
import seaborn as sns
from tqdm import tqdm
import time
np.random.seed(1)

# Preprocess
N_Train_Reduced = 10000    # 10000
N_Train_Full = 60000     # 50000
N_Test = 10000            # 10000

BS = 2000 # batch size


def shuffle_data(X, y):
    shuffle_idx = np.random.permutation(y.size)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    return X, y


def select_balanced_subset(images, labels, use_num_images):
    '''
    select equal number of images from each classes
    '''
    num_total, H, W, C = images.shape
    num_class = np.unique(labels).size
    num_per_class = int(use_num_images / num_class)

    # Shuffle
    images, labels = shuffle_data(images, labels)

    selected_images = np.zeros((use_num_images, H, W, C))
    selected_labels = np.zeros(use_num_images)

    set_id = 0
    sets = {

    }
    for j in range(5):
        for i in range(num_class):
            selected_images[i * num_per_class:(i + 1) * num_per_class] = images[labels == i][j*num_per_class:(j+1)*num_per_class]
            selected_labels[i * num_per_class:(i + 1) * num_per_class] = np.ones((num_per_class)) * i
        # Shuffle again
        selected_images, selected_labels = shuffle_data(selected_images, selected_labels)
        sets[str(j)] = {
            'X' : selected_images,
            'y' : selected_labels
        }

    # Shuffle again
    # selected_images, selected_labels = shuffle_data(selected_images, selected_labels)

    return sets
def maxpool2d(img):
    # print(img.shape)
    res = np.zeros((img.shape[0]//2, img.shape[1]//2, img.shape[2]))
    # print(res.shape)
    for i in range(0,img.shape[0],2):
        for j in range(0,img.shape[1],2):
            for k in range(img.shape[2]):
                # print(img[i:i+1,j:j+1,k])
                res[i//2,j//2,k] = np.max(img[i:i+2,j:j+2,k])
#                 print(res)
    return res

def Shrink(X, shrinkArg):
    #---- max pooling----
    pool = shrinkArg['pool']
    # TODO: fill in the rest of max pooling
    if pool==2:
        X_pooled = np.zeros((X.shape[0], X.shape[1]//pool, X.shape[2]//pool, X.shape[3]))
        for i in range(X.shape[0]):
            X_pooled[i,:,:,:] = maxpool2d(X[i,:,:,:])
        X = X_pooled
    #---- neighborhood construction
    win = shrinkArg['win']
    stride = shrinkArg['stride']
    pad = shrinkArg['pad']
    ch = X.shape[-1]
    # TODO: fill in the rest of neighborhood construction

    if pad<=0:
        pass
    else:
        X = np.pad(X, ((0,0),(pad,pad),(pad,pad),(0,0)), mode='reflect')
    X = view_as_windows(X, (1,win, win, ch), (1,stride, stride,ch))
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

# example callback function for how to concate features from different hops
def Concat(X, concatArg):
    return X
def get_feat(X, p2, num_layers=3, batch_size=10000):
    feats = []
    for j in tqdm(range(X.shape[0]//batch_size)):
    #   print(X[j*batch_size:(j+1)*batch_size].shape)
      output = p2.transform_singleHop(X[j*batch_size:(j+1)*batch_size],layer=0)
      if num_layers>1:
          for i in range(num_layers-1):
              output = p2.transform_singleHop(output, layer=i+1)
      feats.append(output)
    return feats

def problem2(dataset='mnist', model='hop', set_id=0, th1=0.002, th2 = 0.0005, model_save_name='pixel_hop_0_mnist'):
    if dataset == 'mnist':
        (x_train, y_train), (x_test,y_test) = mnist.load_data()
    else:
        (x_train, y_train), (x_test,y_test) = fashion_mnist.load_data()

    x_train = np.asarray(x_train,dtype='float32')[:,:,:,np.newaxis]
    x_test = np.asarray(x_test,dtype='float32')[:,:,:,np.newaxis]
    y_train = np.asarray(y_train,dtype='int')
    y_test = np.asarray(y_test,dtype='int')
    x_train /= 255.0
    x_test /= 255.0

    # if use only 10000 images train pixelhop
    sets = select_balanced_subset(x_train, y_train, use_num_images=N_Train_Reduced)

    if model=='hop':
        SaabArguments = [
            {'num_AC_kernels': -1, 'needBias': False, 'cw': False},
            {'num_AC_kernels': -1, 'needBias': True, 'cw': False},
            {'num_AC_kernels': -1, 'needBias': True, 'cw': False},
        ]
    else:
        SaabArguments = [
            {'num_AC_kernels': -1, 'needBias': False, 'cw': False},
            {'num_AC_kernels': -1, 'needBias': True, 'cw': True},
            {'num_AC_kernels': -1, 'needBias': True, 'cw': True},
        ]

    ShrinkArguments = [
        {'func': Shrink, 'win': 5, 'stride': 1, 'pad': 2, 'pool': 1},
        {'func': Shrink, 'win': 5, 'stride': 1, 'pad': 0, 'pool': 2},
        {'func': Shrink, 'win': 5, 'stride': 1, 'pad': 0, 'pool': 2},
    ]

    X_train_set, y_train_set = sets[str(set_id)]['X'], sets[str(set_id)]['y']
    start = time.time()
    p2 = Pixelhop(depth=3,
            TH1=th1,
            TH2=th2,
            SaabArgs=SaabArguments,
            shrinkArgs=ShrinkArguments,
            concatArg=None
        ).fit(X_train_set)
    pixel_hop_end = time.time() - start
    p2.save(model_save_name)

    start = time.time()
    train_hop3_feats = get_feat(x_train, p2)
    test_hop3_feats = get_feat(x_test, p2)
    feat_end = time.time() - start

    train_hop3_feats = np.concatenate([train_hop3_feats[0],train_hop3_feats[1],train_hop3_feats[2],train_hop3_feats[3],train_hop3_feats[4],train_hop3_feats[5]], axis=0)
    STD = np.std(train_hop3_feats, axis=0, keepdims=1)
    train_hop3_feats =  train_hop3_feats.squeeze()/STD
    test_hop3_feats = test_hop3_feats[0].squeeze()/STD

    start = time.time()
    clf = xgb.XGBClassifier(n_jobs=-1,
                            objective='multi:softprob',
                            # tree_method='gpu_hist', gpu_id=None,
                            max_depth=6,n_estimators=100,
                            min_child_weight=5,gamma=5,
                            subsample=0.8,learning_rate=0.1,
                            nthread=8,colsample_bytree=1.0)

    clf.fit(train_hop3_feats.squeeze(), y_train_set)
    classifier_end = time.time() - start

    y_test_preds = clf.predict(test_hop3_feats.squeeze())
    score = accuracy_score(y_test_preds, y_test)
    C = confusion_matrix(y_test_preds, y_test)
    build = {
        'score': score, 
        'confusion_matrix': C, 
        'train_feats' : train_hop3_feats,
        'test_feats': test_hop3_feats,
        'model': p2, 
        'classifier': clf,
        'model_name': model_save_name,
        'dataset': dataset,
        'pixelhop_time': pixel_hop_end,
        'feat_time': feat_end,
        'classifier_time': classifier_end
    }
    with open(f'{model_save_name}_build.pkl','wb') as f:
        pickle.dump(build, f)
    print_details(build)
    return build
        
def print_details(build):
    print(f"Build {build['model_name']}")
    print("Accuracy score: ", build['score'])
    print("Confusion matrix: ", build['confusion_matrix'])
    print("Train feat shape: ", build['train_feats'].shape)
    print("Test feat shape: ", build['test_feats'].shape)
    print("Pixel hop time: ", build['pixelhop_time'])
    print("Feat time: ", build['feat_time'])
    print("Classifier time: ", build['classifier_time'])
    cf_matrix = build['confusion_matrix'] / np.sum(build['confusion_matrix'])
    if build['dataset']!='mnist':
        idx_to_class = {
                        0: "T-shirt/Top",
                        1: "Trouser",
                        2: "Pullover",
                        3: "Dress",
                        4: "Coat",
                        5: "Sandal",
                        6: "Shirt",
                        7: "Sneaker",
                        8: "Bag",
                        9: "Ankle Boot"
                        }
    else:
        idx_to_class = {idx:str(idx) for idx in range(10)}

    hm=sns.heatmap(cf_matrix , annot=True,
                    fmt='.2%', linewidths=.9, xticklabels=list(idx_to_class.values()), yticklabels=list(idx_to_class.values()))
    hm.get_figure().savefig(f"{build['model_name']}_confusion.png")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    problem2(dataset='mnist', model='hop', set_id=0, th1=0.002, th2 = 0.001, model_save_name='pixel_hop_0_mnist')
    problem2(dataset='mnist', model='hop', set_id=0, th1=0.001, th2 = 0.001, model_save_name='pixel_hop_0_mnist_th0_001')
    problem2(dataset='mnist', model='hop', set_id=0, th1=0.05, th2 = 0.001, model_save_name='pixel_hop_0_mnist_th0_05')
    problem2(dataset='mnist', model='hop', set_id=0, th1=0.005, th2 = 0.001, model_save_name='pixel_hop_0_mnist_th0_005')

    problem2(dataset='fash_mnist', model='hop', set_id=0, th1=0.002, th2 = 0.001, model_save_name='pixel_hop_0_fashmnist')
    problem2(dataset='fash_mnist', model='hop', set_id=0, th1=0.001, th2 = 0.001, model_save_name='pixel_hop_0_fashmnist_th0_001')
    problem2(dataset='fash_mnist', model='hop', set_id=0, th1=0.05, th2 = 0.001, model_save_name='pixel_hop_0_fashmnist_th0_05')
    problem2(dataset='fash_mnist', model='hop', set_id=0, th1=0.005, th2 = 0.001, model_save_name='pixel_hop_0_fashmnist_th0_005')

    problem2(dataset='mnist', model='hop++', set_id=0, th1=0.002, th2 = 0.001, model_save_name='pixel_hop++_0_mnist')
    problem2(dataset='mnist', model='hop++', set_id=0, th1=0.001, th2 = 0.001, model_save_name='pixel_hop++_0_mnist_th0_001')
    problem2(dataset='mnist', model='hop++', set_id=0, th1=0.05, th2 = 0.001, model_save_name='pixel_hop++_0_mnist_th0_05')
    problem2(dataset='mnist', model='hop++', set_id=0, th1=0.005, th2 = 0.001, model_save_name='pixel_hop++_0_mnist_th0_005')

    problem2(dataset='fash_mnist', model='hop++', set_id=0, th1=0.002, th2 = 0.001, model_save_name='pixel_hop++_0_fashmnist')
    problem2(dataset='fash_mnist', model='hop++', set_id=0, th1=0.001, th2 = 0.001, model_save_name='pixel_hop++_0_fashmnist_th0_001')
    problem2(dataset='fash_mnist', model='hop++', set_id=0, th1=0.05, th2 = 0.001, model_save_name='pixel_hop++_0_fashmnist_th0_05')
    problem2(dataset='fash_mnist', model='hop++', set_id=0, th1=0.005, th2 = 0.001, model_save_name='pixel_hop++_0_fashmnist_th0_005')

    problem2(dataset='mnist', model='hop', set_id=0, th1=0.002, th2 = 0.001, model_save_name='pixel_hop_0_mnist_th20_001')
    problem2(dataset='mnist', model='hop', set_id=0, th1=0.002, th2 = 0.01, model_save_name='pixel_hop_0_mnist_th20_01')
    problem2(dataset='mnist', model='hop', set_id=0, th1=0.002, th2 = 0.005, model_save_name='pixel_hop_0_mnist_th20_005')
  
    problem2(dataset='fashmnist', model='hop', set_id=0, th1=0.002, th2 = 0.001, model_save_name='pixel_hop_0_fashmnist_th20_001')
    problem2(dataset='fashmnist', model='hop', set_id=0, th1=0.002, th2 = 0.01, model_save_name='pixel_hop_0_fashmnist_th20_01')
    problem2(dataset='fashmnist', model='hop', set_id=0, th1=0.002, th2 = 0.005, model_save_name='pixel_hop_0_fashmnist_th20_005')

    problem2(dataset='mnist', model='hop++', set_id=0, th1=0.002, th2 = 0.001, model_save_name='pixel_hop++_0_mnist_th20_001')
    problem2(dataset='mnist', model='hop++', set_id=0, th1=0.002, th2 = 0.01, model_save_name='pixel_hop++_0_mnist_th20_01')
    problem2(dataset='mnist', model='hop++', set_id=0, th1=0.002, th2 = 0.005, model_save_name='pixel_hop++_0_mnist_th20_005')
  
    problem2(dataset='fashmnist', model='hop++', set_id=0, th1=0.002, th2 = 0.001, model_save_name='pixel_hop++_0_fashmnist_th20_001')
    problem2(dataset='fashmnist', model='hop++', set_id=0, th1=0.002, th2 = 0.01, model_save_name='pixel_hop++_0_fashmnist_th20_01')
    problem2(dataset='fashmnist', model='hop++', set_id=0, th1=0.002, th2 = 0.005, model_save_name='pixel_hop++_0_fashmnist_th20_005')


    

    
    
    
    
    
    
    
    
    
