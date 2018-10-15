from GEMRank import GEMRank

DATASET_ADDRESS = 'ml-100k/u.data' #'ml-1m/ratings.dat'
#self.users_size = 6040 #movielens 1M
USER_NUMBER = 943 #movielens 100K
EPOCH_NUMBER = 40
EMBEDDING_SIZE = 100

def find_best_hidden_layer_size(embedding_size,size_range, UPL):
    print("Finding best hidden layer size")
    results = []

    for size in size_range:

        gr = GEMRank(UPL, EMBEDDING_SIZE , size, USER_NUMBER, EPOCH_NUMBER, validation_split=0.05 )
        gr.read_data(DATASET_ADDRESS)
        gr.capture_representaions(embedding_size )
        gr.simple_mlp(gr.train_data, embedding_size)
        #ndcg5 = ml.average_NDCG(ml.validation_data,5, ml.predict_top_n)
        accu = gr.calculate_model_acc(ml.validation_data)[1]

        results.append(accu)
        print("For UPL=%s, with hidden_layer_size=%s accuracy is %s"%(UPL, size, accu))

    return size_range[results.index(max(results))]

def main_by_KNI():
    embeddig_range = [50,60,70,80,90,100,110,120,130,140,150]
    repeat = 2
    upl_test =[10,20,50]
    for UPL in upl_test:

        ndcg5_list = []
        ndcg10_list = []
        for i in range(repeat):
            best_embedding_size = 100 #find_best_embedding_size(embeddig_range, UPL)
            best_hidden_layer_size = find_best_hidden_layer_size(best_embedding_size, [5,10,15,20,25], UPL)
            #print("Best Embedding Size is %s"%best_embedding_size)
            print("Best Hidden Layer Size is %s"%best_hidden_layer_size)
            ml = movielens(UPL, best_embedding_size, best_hidden_layer_size )
            print("Accuracy on train: %s"%ml.calculate_model_acc(ml.train_data))
            print("Acuracy on test: %s"%ml.calculate_model_acc(ml.test_data))

            ndcg5 = ml.average_NDCG(ml.test_data,5, ml.predict_top_n_using_KNI)
            ndcg10 = ml.average_NDCG(ml.test_data, 10, ml.predict_top_n_using_KNI)
            #print("#%s average NDCG  is %s"%(i,ndcg))
            ndcg5_list.append(ndcg5)
            ndcg10_list.append(ndcg10)
        ndcg5_avg = (sum(ndcg5_list)/repeat)
        ndcg10_avg = (sum(ndcg10_list) / repeat)
        temp = 0
        for ndcg in ndcg5_list:
            temp += math.pow(ndcg - ndcg5_avg , 2)
        std5 = math.sqrt(temp/repeat)
        temp = 0
        for ndcg in ndcg10_list:
            temp += math.pow(ndcg - ndcg10_avg, 2)
        std10 = math.sqrt(temp / repeat)

        print("UPl: %s "%UPL)
        print("Average in NDCG@5 is %s"%ndcg5_avg)
        print("Standard Deviation is %s"%std5)

        print("Average in NDCG@10 is %s" % ndcg10_avg)
        print("Standard Deviation is %s" % std10)

        output_file = 'result-nn-%s-UPL-%s.txt'%(DATASET_ADDRESS[:DATASET_ADDRESS.index('/')],UPL)

        with open(output_file , 'w') as output:
            output.write("Average in NDCG@5 is %s" % ndcg5_avg)
            output.write("Standard Deviation is %s" % std5)

            output.write("Average in NDCG@10 is %s" % ndcg10_avg)
            output.write("Standard Deviation is %s" % std10)

def main():
    embeddig_range = [50,60,70,80,90,100,110,120,130,140,150]
    repeat = 1
    upl_test =[10,20,50]
    for UPL in upl_test:

        ndcg5_list = []
        ndcg10_list = []
        for i in range(repeat):
            best_hidden_layer_size = 10 #find_best_hidden_layer_size(best_embedding_size, [5,10,15,20,25], UPL)

            print("Best Hidden Layer Size is %s"%best_hidden_layer_size)
            # Set and train GEMRank Model
            gr = GEMRank(UPL, EMBEDDING_SIZE, best_hidden_layer_size, USER_NUMBER ,EPOCH_NUMBER)
            gr.read_data(DATASET_ADDRESS)
            gr.capture_representaions(EMBEDDING_SIZE )
            gr.simple_mlp(gr.train_data, EMBEDDING_SIZE)

            print("Accuracy on train: %s"%gr.calculate_model_acc(gr.train_data))
            print("Acuracy on test: %s"%gr.calculate_model_acc(gr.test_data))

            ndcg5 = gr.average_NDCG(gr.test_data,5, gr.predict_top_n)
            ndcg10 = gr.average_NDCG(gr.test_data, 10, gr.predict_top_n)
            #print("#%s average NDCG  is %s"%(i,ndcg))
            ndcg5_list.append(ndcg5)
            ndcg10_list.append(ndcg10)
        ndcg5_avg = (sum(ndcg5_list)/repeat)
        ndcg10_avg = (sum(ndcg10_list) / repeat)
        temp = 0
        for ndcg in ndcg5_list:
            temp += math.pow(ndcg - ndcg5_avg , 2)
        std5 = math.sqrt(temp/repeat)
        temp = 0
        for ndcg in ndcg10_list:
            temp += math.pow(ndcg - ndcg10_avg, 2)
        std10 = math.sqrt(temp / repeat)

        print("UPl: %s "%UPL)
        print("Average in NDCG@5 is %s"%ndcg5_avg)
        print("Standard Deviation is %s"%std5)

        print("Average in NDCG@10 is %s" % ndcg10_avg)
        print("Standard Deviation is %s" % std10)

        output_file = 'result-useritem-%s-UPL-%s.txt'%(DATASET_ADDRESS[:DATASET_ADDRESS.index('/')],UPL)

        with open(output_file , 'w') as output:
            output.write("Average in NDCG@5 is %s" % ndcg5_avg)
            output.write("Standard Deviation is %s" % std5)

            output.write("Average in NDCG@10 is %s" % ndcg10_avg)
            output.write("Standard Deviation is %s" % std10)

if __name__ == '__main__':
    main()
