import sys;
sys.path.append('../../WorldModelsExperiments/carracing');
sys.path.append('../../WorldModelsExperiments/carracing');
import dream_model
model = dream_model.Model()
import numpy as np
import time
cur_time = time.time()
z,_,_ = model.encode_obs(np.random.rand(64,64,3)*255)
print(z, time.time()-cur_time)
