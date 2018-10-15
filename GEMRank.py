from __future__ import division

import random
import numpy as np
import math
import operator

from multiprocessing import Pool

from sklearn.decomposition import NMF

from keras.layers import Dense , Input , concatenate, Dropout
from keras.models import Model
from keras.utils import plot_model

USER = 0
MOVIE = 1
RATE = 2

class GEMRank():

    def __init__(self, UPL , embedding_size, hidden_layer_size,
                                            users_size,
                                            epoch_number,
                                            user_user = False,
                                            validation_split = 0):

        self.validation_split = validation_split
        self.upl = UPL
        self.user_user = user_user
        self.users_size = users_size
        self.epoch_number = epoch_number
        self.hidden_layer_size = hidden_layer_size

    def set_data(self, train, test, valid):
        self.train_data = train
        self.test_data = test
        self.validation_data = valid

    def set_neural_model(self, model):
        self.model = model

    def read_data(self, dataset_address):

        data = []
        with open(dataset_address) as file:
            for line in file:
                line = line.rstrip('\n')
                if 'ml-1m' in dataset_address:
                    line = line.split('::')
                else:
                    line = line.split()
                line[0] = int(line[0])
                #line[1] = int(line[1])
                line[2] = int(line[2])
                data.append(line)

        np.random.shuffle(data)
        user_included_movies = np.zeros(self.users_size)

        for line in data:
            user_included_movies[line[USER] - 1] += 1

        self.valid_users = np.where(user_included_movies > self.upl + 10)[0]
        #print(self.valid_users)
        user_included_movies = np.zeros(self.users_size)
        user_included_movies_in_validation = np.zeros(self.users_size)
        train_users = []
        train_items = []
        test_users = []
        test_items = []

        train_data = []
        test_data = []
        validation_data = []

        for line in data:
            if line[USER]-1 in self.valid_users:
                if user_included_movies[line[USER]-1] < self.upl:
                    train_data.append(line)
                    user_included_movies[line[USER]-1] += 1

                    train_users.append(line[USER])
                    train_items.append(line[MOVIE])

                #elif user_included_movies_in_validation[line[USER] -1 ] < 10:
                #    validation_data.append(line)
                #    user_included_movies_in_validation[line[USER] -1 ] += 1
                else:
                    test_data.append(line)

                    test_users.append(line[USER])
                    test_items.append(line[MOVIE])
        if self.validation_split > 0:
            np.random.shuffle(train_data)
            validation_size = len(train_data)//20
            for i in range( validation_size ):
                validation_data.append(train_data.pop())

        print("Train Users: %s"%len(set(train_users)))
        print("Train Items: %s" % len(set(train_items)))

        print("Test Users : %s"%len(set(test_users)))
        print("Test Items : %s" % len(set(test_items)))
        self.set_data(train_data, test_data , validation_data)

    def capture_representaions(self, embedding_size,
                                    pco_matrix=None,
                                    dictionary=None ):

        if self.user_user:
            self.MF_model, self.MF_dictionary = self.fit_user_user_MF(
                                                            embedding_size,
                                                            pco_matrix,
                                                            dictionary)
        else:
            self.MF_model, self.MF_dictionary = self.fit_MF(embedding_size,
                                                            pco_matrix,
                                                            dictionary)

        self.user_averages = self.calculate_user_averages(self.train_data)
        if self.user_user:
            self.item_vectors = self.infer_item_vectors(self.train_data)
        else:
            self.user_vectors = self.infer_user_vector( self.train_data, self.user_averages)

    def fit_MF(self, embedding_size, pco_matrix = None, dictionary = None):
        if not pco_matrix and not dictionary:
            sentences = [[] for x in range(self.users_size)]
            movie_dictionary = {}
            movie_index = 0
            for line in self.train_data:
                if line[MOVIE] not in movie_dictionary.keys():
                    movie_dictionary[line[MOVIE]] = movie_index
                    movie_index += 1
                sentences[line[USER] -1 ].append((line[MOVIE],line[RATE]))

            pco_matrix = np.zeros((movie_index, movie_index))

            for line in sentences:
                for movie1 in line:
                    for movie2 in line:
                        if movie1[0] != movie2[0]:
                            pco_matrix[movie_dictionary[movie1[0]], movie_dictionary[movie2[0]]] += 1
            dictionary = movie_dictionary

        for i in range(len(pco_matrix[:,0])):
            for j in range(len(pco_matrix[0,:])):
                if pco_matrix[i,j] != 0 and pco_matrix[i,j] != 1:
                    pco_matrix[i,j] = math.log(pco_matrix[i,j] , 2 )


        total, nonzeros = 0,0
        total = pco_matrix.shape[0] * pco_matrix.shape[1]
        nonzeros =  (pco_matrix > 0).sum()

        #this one
        model = NMF(n_components=embedding_size, init='random', random_state=0, max_iter=1000, tol=2e-4)
        W = model.fit_transform(pco_matrix)
        return W  , dictionary

       # model = NMF(n_components=EMBEDDING_VOL, init='random', random_state=0)
        #return model.fit_transform(item_item_similarity) +  model.components_.T , dictionary

    def fit_user_user_MF(self, embedding_size, pco_matrix = None, dictionary = None):
        if not pco_matrix and not dictionary:

            movie_profiles = {}
            user_dictionary = {}
            user_index = 0
            for line in self.train_data:
                if line[USER] not in user_dictionary.keys():
                    user_dictionary[line[USER]] = user_index
                    user_index += 1
                if line[MOVIE] not in movie_profiles.keys():
                    movie_profiles[line[MOVIE]] = [line[USER]]
                else:
                    movie_profiles[line[MOVIE]].append(line[USER])
            pco_matrix = np.zeros((user_index, user_index))

            for movie in movie_profiles.keys():
                for user1 in movie_profiles[movie]:
                    for user2 in movie_profiles[movie]:
                        if user1 != user2:
                            pco_matrix[user_dictionary[user1], user_dictionary[user2]] +=1

            dictionary = user_dictionary

        for i in range(len(pco_matrix[:,0])):
            for j in range(len(pco_matrix[0,:])):
                if pco_matrix[i,j] >1:
                    pco_matrix[i,j] = math.log(pco_matrix[i,j] , 2 )


        model = NMF(n_components=embedding_size, init='random', random_state=0)
        W = model.fit_transform(pco_matrix)

        return W, dictionary

    def _get_item_vector(self,item):
        if self.user_user:
            return self.item_vectors[item]
        else:
            return self.MF_model[self.MF_dictionary[item],:]

    def _get_user_vector(self,user):
        if self.user_user:
            return self.MF_model[self.MF_dictionary[user+1],:]
        else:
            return self.user_vectors[user]

    def calculate_user_averages(self,data):
        user_averages = [ 0 for x in range(self.users_size)]
        user_counts = [ 0 for x in range(self.users_size)]
        for line in data:
            user_averages[line[USER] -1 ] += line[RATE]
            user_counts[line[USER] -1] += 1
        for i in range(self.users_size):
            if i in self.valid_users and user_counts[i] > 0:
                user_averages[i] /= user_counts[i]
        return user_averages

    def infer_user_vector(self, data, user_averages):
        user_vectors = []

        for i in range(self.users_size):
            if i in self.valid_users:
                y=0
                uv = 0
                for line in data:
                    if line[USER]-1 == i:
                        y += 1
                        uv += ((line[RATE] - user_averages[i]) * self._get_item_vector(line[MOVIE]))

                user_vectors.append(uv/y)
            else:
                user_vectors.append(0)
        return user_vectors

    def infer_item_vectors(self, data):
        item_vectors = {}
        for line in data:
            item_vectors[line[MOVIE]] = 0

        for i in item_vectors.keys():
            y=0
            iv = 0
            for line in data:
                if line[MOVIE] == i:

                    iv +=  self._get_user_vector(line[USER]-1)
                    y += 1

            item_vectors[i] = iv/y

        return item_vectors

    def simple_mlp(self, data, embedding_size):

        mlp_data_user, mlp_data_item, targets = self._get_data_for_deep_model(data)

        user_inputs = Input(shape=(embedding_size,))
        item_inputs = Input(shape=(embedding_size,))
        added = concatenate([user_inputs,item_inputs])
        #dropout1 = Dropout(0.2)(added)
        shared_layer = Dense(self.hidden_layer_size, activation='relu')(added)
        #final_layer = Dense(5, activation='relu')(shared_layer)
        dropout2 = Dropout(0.5)(shared_layer)
        prediction = Dense(1, activation='sigmoid')(dropout2)

        model = Model(inputs=[user_inputs, item_inputs], outputs=prediction)
        model.summary()
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop'
                      , metrics=['accuracy']
                      )

        model.fit([mlp_data_user, mlp_data_item], targets,
                                                epochs=self.epoch_number,
                                                verbose=False)

        self.set_neural_model(model)

    def _get_data_for_deep_model(self, data):
        mlp_data_user = []
        mlp_data_item = []
        targets = []
        for line in data:

            mlp_data_item.append(self._get_item_vector(line[MOVIE]))
            mlp_data_user.append(self._get_user_vector(line[USER] -1))
            if line[RATE] > self.user_averages[line[USER] -1]:
                targets.append(1)
            else:
                targets.append(0)
            #targets.append(line[RATE]/5)
        mlp_data_item = np.array(mlp_data_item)
        mlp_data_user = np.array(mlp_data_user)
        targets = np.array(targets)

        return mlp_data_user ,mlp_data_item , targets

    def predict_top_n(self,data,  user, n):
        added_movies_to_test_data = []
        user_vector = self._get_user_vector(user)
        mlp_data_user = []
        mlp_data_item = []
        mlp_targets = []
        for line in data:
            if line[USER] - 1  == user:
                movie = line[MOVIE]
                try:
                    mlp_data_item.append(self._get_item_vector(line[MOVIE]))
                    mlp_data_user.append(user_vector)
                    mlp_targets.append(line[RATE])
                    added_movies_to_test_data.append(movie)
                except:
                    #print("no representation for movie %s"%movie)
                    pass
        if len(mlp_data_item) < n:
            return []
        mlp_data_user = np.array(mlp_data_user)
        mlp_data_item = np.array(mlp_data_item)

        #mlp_data_user = mlp_data_user.reshape( (-1,EMBEDDING_VOL,1) )
        #mlp_data_item = mlp_data_item.reshape( (-1,EMBEDDING_VOL,1) )

        #print("input_1 size %s"%len(mlp_data_user))
        #print("input_2 size %s"%len(mlp_data_item))
        added_movies_to_test_data = np.array(added_movies_to_test_data)
        predicted =  self.model.predict([mlp_data_user, mlp_data_item])

        predicted = predicted.flatten()
        predicted_args = predicted.argsort()
        top_n = predicted_args[-n:]
        top_n =  np.flip(top_n , 0)

        #eval_list = self.model.evaluate(x=[mlp_data_user,mlp_data_item] , y=mlp_targets , batch_size=1000 , verbose=0)

        return added_movies_to_test_data[top_n]

    def predict_top_n_using_KNI(self,data,  user, n):

        added_movies_to_test_data = []
        for line in data:
            if line[USER] - 1  == user:
                movie = line[MOVIE]
                added_movies_to_test_data.append(movie)

        added_movies_to_test_data = np.array(added_movies_to_test_data)
        predicted = []
        for m in added_movies_to_test_data:
            try:
                predicted.append(self._cosin_similarity( self._get_user_vector(user) ,
                                                    self._get_item_vector(m)))
            except:
                #print("no representation for movie %s"%m)
                pass
        predicted = np.array(predicted)
        predicted_args = predicted.argsort()
        top_n = predicted_args[-n:]
        top_n =  np.flip(top_n , 0)
        return added_movies_to_test_data[top_n]

    def _euclidean_distance(self, v1, v2):
        return math.sqrt(sum(np.square(v1-v2)))
    def _cosin_similarity(self, v1, v2):
        return (1 - spatial.distance.cosine(v1, v2))

    def calculate_NDCG(self, data, user, top_ranked, n):
        top_ranked_rates = []
        for top_ranked_movie in top_ranked:
            for line in data:
                if line[USER] -1 == user:
                    movie = line[MOVIE]
                    if movie == top_ranked_movie:
                        top_ranked_rates.append(line[RATE])
                        break
        all_user_rates = []
        for line in data:
            if line[USER] -1 == user:
                all_user_rates.append(line[RATE])
        all_user_rates.sort(reverse=True)
        top_real_rates = all_user_rates[0:n]

        p_u = self.dcg(top_ranked_rates)
        beta_u = self.dcg(top_real_rates)

        return p_u/beta_u

    def dcg(self, l):
        dcg = 0
        for idx in range(len(l)):
            dcg += (2**l[idx] - 1) / (math.log( (idx+2) ,2) )
        return dcg


    def calc(self, user, data,  n, predictor):

        top_n = predictor(data, user,n)
        if len(top_n) < n:
            #print("user %s NDCG is: NaN" %(user))
            return -1
        else:
            ndcg = self.calculate_NDCG(data, user, top_n, n)
            #print("user %s NDCG is: %s" %(user, ndcg))
            return ndcg



    def average_NDCG(self, data,  n, predictor):

        #print("calculating average NDCG of %s users..." %self.users_size )

        #NDCGs = Parallel(n_jobs=2)(delayed(self.calc)(user, data, n, predictor) for user in range(self.users_size))
        p = Pool(2)
        #NDCGs = p.map(self.calc, [(user, data,  n, predictor) for user in range(self.users_size) if user in self.valid_users])
        NDCGs = [self.calc(user,data, n , predictor) for user in range(self.users_size) if user in self.valid_users]
        valids = [n for n in NDCGs if n != -1]
        return sum(valids)/len(valids)

    def calculate_model_acc(self, data):
        users = []
        items = []
        targets =[]

        for line in data:
            try:
                items.append(self._get_item_vector(line[MOVIE]))
                users.append(self._get_user_vector(line[USER] -1))
                if line[RATE] > self.user_averages[line[USER] - 1]:
                    targets.append(1)
                else:
                    targets.append(0)
                #targets.append(line[RATE]/5)
            except KeyError:
                pass

        print("%s %s"%(len(users) , len(items)))
        return self.model.evaluate([users, items], y=targets)
