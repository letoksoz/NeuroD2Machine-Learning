using Knet
using FileIO
#################################PrepareData#####################
function balancedIndexList(y,window_size)
    trueIndexList = find(y)
    falseIndexList = find(.!y)

    if length(falseIndexList) > length(trueIndexList)
        shuffledIndexes = randperm(length(falseIndexList))
        subSuffledIndexes = shuffledIndexes[1:length(trueIndexList)]
        falseIndexList = falseIndexList[subSuffledIndexes]
    elseif length(trueIndexList) > length(falseIndexList)
        shuffledIndexes = randperm(length(trueIndexList))
        subSuffledIndexes = shuffledIndexes[1:length(falseIndexList)]
        trueIndexList = trueIndexList[subSuffledIndexes]
    end

    indexList = sort(vcat(trueIndexList,falseIndexList))

    indexList = indexList[randperm(length(indexList))]

    window_shift = Int((window_size - 1) / 2)

    return indexList[window_shift:end-window_shift]
end

function generateWindowedData(x,y,indexList,window_size)
    sample_size = length(x[1,:])
    #Adding Window with window size 2*window_shift+1
    window_shift = Int((window_size - 1) / 2)

    indexList_x = indexList .- window_shift

    windows = Array{Float32}(window_size,1,15,length(indexList))

    for n = -window_shift:window_shift
        m = x[:,(window_shift+1+n):(sample_size-window_shift+n)]
        #push!(windows[n+window_shift+1,:,:],  m[:,indexList_x])
        windows[n+window_shift+1,1,:,:] = m[:,indexList_x]
    end
    y = y[:,indexList]
    return windows, y
end

function prepareData(fileprefix,window_size)
    ##Class 1---Without NeuroD2, Class 2---With NeuroD2
    data = load("levent.jld2","data")

    y=data[:,1]
    x=data[:,2:end]

    balanced_index_list = balancedIndexList(y,window_size)

    chr_data =  transpose(x)
    y_data = transpose(y)

    windowed_chr_data,y_data = generateWindowedData(chr_data,y_data,balanced_index_list,window_size)

    y_data = Array{Int64}(y_data)
    y_data = 1 + vec(y_data)


    #Creating Shuffled IndexList
    num_samples = length(balanced_index_list)
    num_trn_samples = Int(floor(num_samples * 0.9))
    num_val_samples = Int(floor((num_samples - num_trn_samples - 2) / 2))
    num_test_samples = num_val_samples
    indexes = randperm(num_samples)
    trn_index_list = indexes[1:num_trn_samples]
    val_index_list = indexes[num_trn_samples+1:num_trn_samples+num_val_samples]
    test_index_list = indexes[num_val_samples+num_trn_samples+1:num_test_samples+num_val_samples+num_trn_samples]

    windowed_chr_trn = windowed_chr_data[:,:,:,trn_index_list]
    y_trn = y_data[trn_index_list]
    windowed_chr_val = windowed_chr_data[:,:,:,val_index_list]
    y_val = y_data[val_index_list]
    windowed_chr_test = windowed_chr_data[:,:,:,test_index_list]
    y_test = y_data[test_index_list]

    global dtrn = Any[]
    for i = 1:100:(length(y_trn)-100)
        push!(dtrn, (windowed_chr_trn[:,:,:,i:i+99],y_trn[i:i+99]))
    end

    global dtst = Any[]
    for i = 1:100:(length(y_test)-100)
        push!(dtst, (windowed_chr_test[:,:,:,i:i+99],y_test[i:i+99]))
    end
end
####################################################################

#############################ML_Functions###########################
function sl()
    println("******************************************************************")
end

#=
The trained architecture has the form of:
conv -> conv -> conv ->  Avg.pool -> fc -> output
=#

function predict_CNN(w,x)
    # conv. layer
    x = relu.(conv4(w[1], x; padding=(2,1), stride=(2,1)) .+ w[2])
    # conv. layer
    x = relu.(conv4(w[3], x; padding=(2,1), stride=(2,1)) .+ w[4])
    # conv. layer
    x = relu.(conv4(w[5], x; padding=(2,1), stride=(2,1)) .+ w[6])
    #global avg.pool layer
    x = pool(x,window=7,mode=1)
    #fc layer
    x = relu.(w[7] * mat(x) .+ w[8])
    return w[9] * mat(x) .+ w[10]
end

loss_CNN(w,x,y_actual) = nll(predict_CNN(w,x),y_actual)
lossgradient_CNN = grad(loss_CNN)

function train(w, dtrn; lr=.01, epochs=10)
    optim = optimizers(w, Adam; lr=lr)
    println("Starting the epoch 1:")
    for epoch=1:epochs
        for (x,y) in dtrn
            x = KnetArray{Float32}(x)
            g = lossgradient_CNN(w, x, y)
            update!(w,g,optim)
        end

        if (epoch % 3) == 1
            println("epoch ",epoch," is over")
        end

    end
    return w
end

function initialize_weight_CNN()
    weight = [ xavier(Float32,5,1,15,64),  zeros(Float32,1,1,64,1),
               xavier(Float32,5,1,64,64), zeros(Float32,1,1,64,1),
               xavier(Float32,5,1,64,64),  zeros(Float32,1,1,64,1),
               xavier(Float32,512,64),  zeros(Float32,512,1),
               xavier(Float32,2,512),  zeros(Float32,2,1)]
    return map(KnetArray{Float32}, weight)
end

function print_weight(message, weight)
    weight = map(Array{Float32}, weight)
    println(message)
    for i=1:2:length(weight)
        sl()
        println("Weights Values")
        println(weight[i])
        println("Bias Values")
        println(weight[i+1])
        sl()
    end
end
####################################################################

#############################Main################################
fileprefix = "chr1"
batchsize = 100

#window_size_list = [1,3,5,7,9,11,13,15,17,19,25,31,39,45,51,55,61,67]
#trn_accuracy_values = Any[]
#tst_accuracy_values = Any[]
window_size = 11
prepareData(fileprefix,window_size)
w = initialize_weight_CNN()
w = train(w, dtrn; lr=.01, epochs=10)
accuracy(w,dtrn,predict_CNN)

push!(trn_accuracy_values, accuracy(w,dtrn,predict))
push!(tst_accuracy_values, accuracy(w,dtst,predict))
println("Training and Test Accuracy Values Are Calculated for the Window Size:",1)


trn_accuracy_values
tst_accuracy_values
#################################################################
#trn_loss
