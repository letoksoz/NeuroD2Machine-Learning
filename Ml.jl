workspace()
using Knet

#NOTLAR
#KODU BASITLESTIR
#MUMKUNSE TRANSPOSELARDAN KURTAR
#enum VARIABLE_TYPE {
#		ENCFF145FVU, ENCFF091JOV, ENCFF102IIL, ENCFF676DBG, ENCFF875CQU, ENCFF152TUF, A, T, G, C, NEUROD2, TSSupstream, CDS, INTRON, UTR5, UTR3;
#	}

#Reading the Data
fileprefix = "chr1"
rows = readdlm(string(fileprefix , "num_rows.txt"))
num_rows = convert(Int,rows[1])
columns = readdlm(string(fileprefix , "num_columns.txt"))
num_columns = convert(Int,columns[1])
num_samples = num_columns
num_parameters = num_rows - 1

bitarraydata = falses(num_rows,num_columns)
chrdata = read!(string(fileprefix , "bitarray.txt")::AbstractString, bitarraydata::Union{Array, BitArray})

inputarraydata = falses(num_rows,num_columns)
input_data = transpose(read!(string(fileprefix , "inputdataFINAL.txt")::AbstractString, inputarraydata::Union{Array, BitArray}))

#Defining the fuctions
predict(w,x) = Knet.sigm(x*w)

function loss(w,x,y,batch_range)
    batch_size = maximum(batch_range) - minimum(batch_range)
    x_converted = KnetArray{Float32}(x[batch_range,:])
    #log. return regular array not Knet Array
    result = -(1. / batch_size) * (transpose(y[batch_range,1:1]) * KnetArray{Float32}(log.(predict(w,x_converted))) + (1 .- transpose(y[batch_range,1:1])) * KnetArray{Float32}(log.(1 .- predict(w,x_converted))))
    return result
end

#added 0 ----take the derivative with respect to row
#lossgradient = grad(loss)
function lossgradient(w,x,y,batch_range)
    batch_size = maximum(batch_range) - minimum(batch_range)
    x_converted = KnetArray{Float32}(x[batch_range,:])
    return (1. / batch_size) .* (transpose(x_converted) * (predict(w,x_converted) - y[batch_range,1:1]))
end

function mini_batch_rangeFinder(x,y,n)
    size = Int(floor(num_samples / n) + 1)
    last_batch_size = num_samples % n
    array_of_ranges = Array{UnitRange{Int64}}(size-1)
    starting_point = 1
    ending_point = n
    for nm = 1:size-1
        array_of_ranges[nm] = starting_point:ending_point
        starting_point += n
        ending_point +=n
    end
    #array_of_ranges[size] = starting_point+1:(ending_point-n+last_batch_size)
    return array_of_ranges
end

#Calculating the Accuracy
#Reading the Test Data
rows_test = readdlm("chr2num_rows.txt")
num_rows_test = convert(Int,rows_test[1])
columns_test = readdlm("chr2num_columns.txt")
num_columns_test = convert(Int,columns_test[1])

bitarraydata_test = falses(num_rows_test,num_columns_test)
chrdata_test = read!("chr2bitarray.txt"::AbstractString, bitarraydata_test::Union{Array, BitArray})

bitarraytest_input = falses(num_rows_test,num_columns_test)
test_input = transpose(read!("test_input.txt"::AbstractString, bitarraytest_input::Union{Array, BitArray}))

y_test = KnetArray{Float32}(reshape((chrdata_test[11,:]),(num_columns_test,1)))

#computes the accuracy per batch and total accuracy
function accuracy(w,test_sample_input,test_sample_result,test_batch_range)
    threshold = 0.5
    predicted_result = 0
    num_right_guess = 0
    num_wrong_guess = 0
    xtest_converted = KnetArray{Float32}(test_sample_input[test_batch_range,:])
    result_vector = predict(w,xtest_converted)
    for num_test in test_batch_range
        if (result_vector[num_test,1] > threshold)
            predicted_result = 1
        end

        if (test_sample_result[num_test,1] == predicted_result)
            num_right_guess += 1
        else
            num_wrong_guess += 1
        end
    end
    return float(num_right_guess) / (num_right_guess + num_wrong_guess)
end


#PREPARING THE DATA FOR TRAINING
#setting y array
y = KnetArray{Float32}(reshape(chrdata[11,:],(num_columns,1)))

#setting up the weight vectors
rng = MersenneTwister(1234)
w = KnetArray{Float32}(rand!(rng, zeros(num_rows,1)))

#SILINECEK
for i=1:16
    println(w[i,1])
end

#Training the sample
#rowu toplamadim
numepochs = 40
alpha = 0.1
minibatch_list = mini_batch_rangeFinder(input_data,y,256)
#[3000000:3001000
Optimization_Algorithm = Sgd(lr=alpha)
for epoch=1:numepochs
    for range_instance in minibatch_list
        g = lossgradient(w, input_data, y,range_instance)
        #stochastic gradient descent
        update!(w, g, Optimization_Algorithm)
        #update!(w, g, Adam(lr=alpha))
        #w = w - lr * g

        #for i in 1:length(w)
        #        w[i] = w[i] - lr * g[i]
        #end
    end
    if epoch == 1
        println("1inci epoch bitti")
    end
    if epoch == 2
        println("2inci epoch bitti")
    end
    if epoch == 5
        println("5inci epoch bitti")
    end
    if epoch == 10
        println("10inci epoch bitti")
    end
    if epoch == 15
        println("15inci epoch bitti")
    end
    if epoch == 20
        println("20inci epoch bitti")
    end
end

w

#WORK IN PROGRESS
for test_range_instance in minibatch_list
    accuracy(w,test_input,y_test,test_range_instance)
end

#SILINECEK
for i=1:16
    println(w[i,1])
end
