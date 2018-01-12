#Accuracy Module

function prepare_raw_test_data()
    #Loading Test Data - Using chr-2 as a test data
    fileprefix_test = "chr2"
    rows_test = readdlm(string(fileprefix_test , "num_rows.txt"))
    num_rows_test = convert(Int,rows_test[1])
    columns_test = readdlm(string(fileprefix_test , "num_columns.txt"))
    num_columns_test = convert(Int,columns_test[1])

    bitarraydata_test = falses(num_rows_test,num_columns_test)
    chrdata_test = read!(string(fileprefix_test , "bitarray.txt")::AbstractString, bitarraydata_test::Union{Array, BitArray})

    #setting y array
    y_chr2 = reshape(chrdata_test[1,:],(num_columns_test,1))

    #Switch the result row with row of 1`s for the bias term in weight
    chrdata_test[1,:] = fill!(chrdata_test[1,:], true)

    #now columns represent parameters and rows represent samples
    x_trn_test = transpose(chrdata_test)

    test_list = mini_batch_rangeFinder(x_trn_test,y_chr2,256,num_columns_test)
    return test_list,y_chr2,x_trn_test
end

function prepare_balanced_test_data(fileprefix)
    #Creating and Reading the (Balanced)Test Data for chr1
    test_fileName = string("test_", fileprefix, "_indexes.txt")
    test_balanced_index_list = readdlm(test_fileName)

    #Index starts at 1 in Julia
    test_balanced_index_list = Int.(test_balanced_index_list .+ 1)
    return test_balanced_index_list
end

type Precision_Recall
    TN
    TP
    FN
    FP
end

function accuracy(actual_vector,predict_vector, precision_recall)
    threshold = 0.5

    for num_test = 1:length(predict_vector)
        if (predict_vector[num_test,1] > threshold)
            predicted_result = 1

            if (actual_vector[num_test,1] == predicted_result)
                precision_recall.TP += 1
            else
                precision_recall.FP += 1
            end
        else
            predicted_result = 0

            if (actual_vector[num_test,1] == predicted_result)
                precision_recall.TN += 1
            else
                precision_recall.FN += 1
            end
        end

    end
end


function batch_accuracy(y_chr,x_trn,test_range_instance,w_dummy,w,precision_recall_dummy,precision_recall)

    y_actual_vector = y_chr[test_range_instance,1:1]

    xtest_converted = KnetArray{Float32}(x_trn[test_range_instance,:])

    #For the Dummy Weight
    predict_vector = Array{Float32}(predict(w_dummy,xtest_converted))
    accuracy(y_actual_vector,predict_vector,precision_recall_dummy)

    #For the Regular Weight
    predict_vector = Array{Float32}(predict(w,xtest_converted))
    accuracy(y_actual_vector,predict_vector,precision_recall)
end


function printResultStatistics(precision_recall_dummy,precision_recall)
    sl()
    println("Dummy accuracy is:",calculate_accurancy(precision_recall_dummy))
    println("Regular accuracy is:",calculate_accurancy(precision_recall))
    println("Total predicted positive:",precision_recall.TP+precision_recall.FP)
    println("True positive:",precision_recall.TP)
    println("False positive:",precision_recall.FP)


    println("Recall dummy is:",calculate_recall(precision_recall_dummy))
    println("Precision dummy is:",calculate_precision(precision_recall_dummy))


    println("Recall is:",calculate_recall(precision_recall))
    println("Precision is:",calculate_precision(precision_recall))
    sl()

end

function calculate_accurancy(pr)
    return (pr.TP + pr.TN) / (pr.TP + pr.TN + pr.FP + pr.FN)
end

function calculate_recall(pr)
    return pr.TP / (pr.TP + pr.FN)
end

function calculate_precision(pr)
    return pr.TP / (pr.TP + pr.FP)
end
