#Log Regression Module

include("Ml_module.jl")
include("accuracy_module.jl")


function execute_log_regression(fileprefix)
    #Reading the Data

    rows = readdlm(string(fileprefix , "num_rows.txt"))
    num_rows = convert(Int,rows[1])
    columns = readdlm(string(fileprefix , "num_columns.txt"))
    num_columns = convert(Int,columns[1])

    bitarraydata = falses(num_rows,num_columns)
    chrdata = read!(string(fileprefix , "bitarray.txt")::AbstractString, bitarraydata::Union{Array, BitArray})

    #setting y array
    y_chr1 = reshape(chrdata[1,:],(num_columns,1))

    #Switch the result row with row of 1`s for the bias term in weight
    chrdata[1,:] = fill!(chrdata[1,:], true)

    #now columns represent parameters and rows represent samples
    x_trn = transpose(chrdata)

    #PREPARING THE DATA FOR TRAINING

    #setting up the weight vectors
    rng = MersenneTwister(1234)
    w = KnetArray{Float32}(rand!(rng, zeros(num_rows,1)))

    #setting up a dummy weight vectors
    rng_dummy = MersenneTwister(5678)
    w_dummy = KnetArray{Float32}(rand!(rng_dummy, zeros(num_rows,1)))

    print_weight("inital values of weight vector",w)
    print_weight("dummy weight vector",w_dummy)

    #Training the sample
    numepochs = 40
    alpha = 0.001
    num_samples_per_batch = 256

    #For regular batching
    minibatch_list = mini_batch_rangeFinder(x_trn,y_chr1,num_samples_per_batch,num_columns)

    #For balanced batching
    trn_fileName = string("training_", fileprefix, "_indexes.txt")
    trn_index_list = readdlm(trn_fileName)

    #Index starts at 1 in Julia
    trn_index_list = Int.(trn_index_list .+ 1)

    #stochastic gradient descent
    Optimization_Algorithm = Adam(lr = alpha)
    #Sgd(lr=alpha)
    for epoch=1:numepochs

        #Balanced Batching
        for nm=1:length(trn_index_list[:,1])
            range_instance = trn_index_list[nm,:]
            g = lossgradient(w, x_trn, y_chr1,range_instance,num_samples_per_batch)
            update!(w, g, Optimization_Algorithm)
        end

        if (epoch % 3) == 1
            println("epoch ",epoch," is over")
        end
    end

    print_weight("final values of weight vector",w)


    #Balanced Test Data
    test_balanced_index_list = prepare_balanced_test_data(fileprefix)

    precision_recall_dummy = Precision_Recall(0,0,0,0)
    precision_recall = Precision_Recall(0,0,0,0)

    for nm_test=1:length(test_balanced_index_list[:,1])
        test_range_instance = test_balanced_index_list[nm_test,:]

        batch_accuracy(y_chr1,x_trn,test_range_instance,w_dummy,w,precision_recall_dummy,precision_recall)
    end

    printResultStatistics(precision_recall_dummy,precision_recall)

    #Raw Test Data
    test_list,y_chr2,x_trn_test = prepare_raw_test_data()

    precision_recall_dummy = Precision_Recall(0,0,0,0)
    precision_recall = Precision_Recall(0,0,0,0)

    for test_range_instance in test_list
        batch_accuracy(y_chr2,x_trn_test,test_range_instance,w_dummy,w,precision_recall_dummy,precision_recall)
    end

    printResultStatistics(precision_recall_dummy,precision_recall)

end
