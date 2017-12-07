using Knet

function sl()
    println("******************************************************************")
end

#Reading the Data
fileprefix = "chr1"
rows = readdlm(string(fileprefix , "num_rows.txt"))
num_rows = convert(Int,rows[1])
columns = readdlm(string(fileprefix , "num_columns.txt"))
num_columns = convert(Int,columns[1])

bitarraydata = falses(num_rows,num_columns)
chrdata = read!(string(fileprefix , "bitarray.txt")::AbstractString, bitarraydata::Union{Array, BitArray})

#setting y array
y_chr1 = KnetArray{Float32}(reshape(chrdata[1,:],(num_columns,1)))

#Switch the result row with row of 1`s for the bias term in weight
chrdata[1,:] = fill!(chrdata[1,:], true)

#now columns represent parameters and rows represent samples
x_trn = transpose(chrdata)

#Defining the fuctions
predict(w,x) = Knet.sigm(x*w)

function loss(w,x,y,batch_range,batch_size)
    x_converted = KnetArray{Float32}(x[batch_range,:])
    result = -(1. / batch_size) * (transpose(y[batch_range,1:1]) * KnetArray{Float32}(log.(predict(w,x_converted))) + (1 .- transpose(y[batch_range,1:1])) * KnetArray{Float32}(log.(1 .- predict(w,x_converted))))
    return result
end

#Use lossgradient = grad(loss) to import grad function from the Knet package for complex functions
function lossgradient(w,x,y,batch_range,batch_size)
    x_converted = KnetArray{Float32}(x[batch_range,:])
    return (1. / batch_size) .* (transpose(x_converted) * (predict(w,x_converted) - y[batch_range,1:1]))
end

#regular mini-batching, please use parser.java to import ranges for oversampled mini-batches
function mini_batch_rangeFinder(x,y,n,num_samples)
    #omitting the last chunk
    size = Int(floor(num_samples / n))
    last_batch_size = num_samples % n
    array_of_ranges = Array{UnitRange{Int64}}(size)
    starting_point = 1
    ending_point = n
    for nm = 1:size
        array_of_ranges[nm] = starting_point:ending_point
        starting_point += n
        ending_point += n
    end
    return array_of_ranges
end

#Calculating the Accuracy

#Creating and Reading the (Balanced)Test Data
test_fileName = string("test_", fileprefix, "_indexes.txt")
test_index_list = readdlm(test_fileName)

#Index starts at 1 in Julia
test_index_list = Int.(test_index_list .+ 1)

#computes the accuracy
function accuracy(test_sample_result,result_vector)
    num_right_guess = 0
    num_wrong_guess = 0
    threshold = 0.5

    for num_test = 1:length(result_vector)
        if (result_vector[num_test,1] > threshold)
            predicted_result = 1
        else
            predicted_result = 0
        end

        if (test_sample_result[num_test,1] == predicted_result)
            num_right_guess += 1
        else
            num_wrong_guess += 1
        end
    end
    accuracy_per_batch = float(num_right_guess) / (num_right_guess + num_wrong_guess)
    return num_right_guess,num_wrong_guess
end


#PREPARING THE DATA FOR TRAINING

#setting up the weight vectors
rng = MersenneTwister(1234)
w = KnetArray{Float32}(rand!(rng, zeros(num_rows,1)))

#setting up a dummy weight vectors
rng_dummy = MersenneTwister(5678)
w_dummy = KnetArray{Float32}(rand!(rng_dummy, zeros(num_rows,1)))

#Checking inital values of weight vector
println("inital values of weight vector")
for i=1:16
    println(w[i,1])
end
sl()

#Checking values of dummy weight vector
println("dummy weight vector")
for i=1:16
    println(w_dummy[i,1])
end
sl()

#Training the sample
numepochs = 40
alpha = 0.1
num_samples_per_batch = 256

#For regular batching
minibatch_list = mini_batch_rangeFinder(x_trn,y_chr1,num_samples_per_batch,num_columns)

#For balanced batching
trn_fileName = string("training_", fileprefix, "_indexes.txt")
trn_index_list = readdlm(trn_fileName)

#Index starts at 1 in Julia
trn_index_list = Int.(trn_index_list .+ 1)

#stochastic gradient descent
Optimization_Algorithm = Sgd(lr=alpha)
for epoch=1:numepochs

    #=Regular Batching
    for range_instance in minibatch_list
        g = lossgradient(w, x_trn, y_chr1,range_instance,num_samples_per_batch)
        update!(w, g, Optimization_Algorithm)
    end
    =#

    #Balanced Batching
    for nm=1:length(trn_index_list[1,:])
        range_instance = trn_index_list[nm,:]
        g = lossgradient(w, x_trn, y_chr1,range_instance,num_samples_per_batch)
        update!(w, g, Optimization_Algorithm)
    end

    if (epoch % 3) == 1
        println("epoch ",epoch," is over")
    end
end

g = lossgradient(w, x_trn, y_chr1,,256)

#Cheking the final weight values
println("final values of weight vector")
for i=1:16
    println(w[i,1])
end
sl()

#Total Accuracy
dummy_total_right = 0
dummy_total_wrong = 0

regular_total_right = 0
regular_total_wrong = 0

testbatch_list = mini_batch_rangeFinder(x_test,y_chr2,num_samples_per_batch,num_columns_test)


for test_range_instance in testbatch_list

    xtest_converted = KnetArray{Float32}(x_test[test_range_instance,:])
    result_vector = Array{Float32}(predict(w_dummy,xtest_converted))

    dummy_per_batch_right,dummy_per_batch_wrong = accuracy(y_chr2,result_vector) #(w_dummy,x_test,y_chr2,test_range_instance,num_samples_per_batch)
    dummy_total_right += dummy_per_batch_right
    dummy_total_wrong += dummy_per_batch_wrong



    xtest_converted = KnetArray{Float32}(x_test[test_range_instance,:])
    result_vector = Array{Float32}(predict(w,xtest_converted))

    regular_per_batch_right,regular_per_batch_wrong = accuracy(y_chr2,result_vector) #(w,x_test,y_chr2,test_range_instance,num_samples_per_batch)
    regular_total_right += regular_per_batch_right
    regular_total_wrong += regular_per_batch_wrong
end
regular_total_accuracy = float(regular_total_right) / (regular_total_right + regular_total_wrong)
dummy_total_accuracy = float(dummy_total_right) / (dummy_total_right + dummy_total_wrong)

println("Regular accuracy is:",regular_total_accuracy)
println("Dummy accuracy is:",dummy_total_accuracy)
