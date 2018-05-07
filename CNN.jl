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
        windows[n+window_shift+1,1,:,:] = m[:,indexList_x]
    end
    y = y[:,indexList]
    return windows, y
end

function prepareData(fileprefix,window_size,batchsize)
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

    global dtrn = minibatch(windowed_chr_trn, y_trn, batchsize;shuffle=true,xtype=KnetArray{Float32})
    global dtst = minibatch(windowed_chr_test, y_test, batchsize;shuffle=true,xtype=KnetArray{Float32})
end
####################################################################

#############################ML_Functions###########################
function sl()
    println("******************************************************************")
end

function conv_layer(w, m, x)
    cl = conv4(w[1], x; padding=(2,1), stride=(2,1))
    cl = batchnorm(cl, m, w[2])
    cl = relu.(cl)
    return cl
end

function fc_layer(w, m, x)
    fc = w[1] * mat(x)
    fc = batchnorm(fc, m, w[2])
    return relu.(fc)
end

#=
The trained architecture has the form of:
conv -> conv -> conv ->  Avg.pool -> fc -> output
=#

function predict(w, m, x)
    # conv. layer
    x = conv_layer(w[1:2], m[1], x)
    # conv. layer
    x = conv_layer(w[3:4], m[2], x)
    # conv. layer
    x = conv_layer(w[5:6], m[3], x)
    #global avg.pool layer
    x = pool(x,window=7,mode=1)
    #fc layer
    x = fc_layer(w[7:8], m[4], x)
    return w[9] * mat(x) .+ w[10]
end

loss(w, m, x, y_actual) = nll(predict(w, m, x), y_actual)
lossgradient = grad(loss)

function train(w, m, dtrn; lr=.01, epochs=10)
    optim = optimizers(w, Adam; lr=lr)
    println("Starting the epoch 1:")
    for epoch=1:epochs
        for (x,y) in dtrn
            g = lossgradient(w, m, x, y)
            update!(w,g,optim)
        end

        if (epoch % 3) == 1
            println("epoch ",epoch," is over")
        end

    end
    return w
end

function initialize_weight()
    weight = [ xavier(Float32,5,1,15,64),  bnparams(Float32, 64),
               xavier(Float32,5,1,64,64), bnparams(Float32, 64),
               xavier(Float32,5,1,64,64),  bnparams(Float32, 64),
               xavier(Float32,512,64),  bnparams(Float32, 512),
               xavier(Float32,2,512),  zeros(Float32,2,1)]
    return map(KnetArray{Float32}, weight)
end

function initialize_model()
    # Initializing weights
    w = initialize_weight()
    # Initializing a moments object/batchnorm
    m = Any[bnmoments() for i = 1:4]
    return w,m
end

# Accuracy computation
function compute_accuracy(w, m, data=dtst)
    model = (w, m)
    return accuracy(model, data, (model, x)->predict(model[1], model[2], x); average=true)
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
window_size = 11

prepareData(fileprefix,window_size,batchsize)
w, m = initialize_model()
w = train(w, m, dtrn; lr=.01, epochs=10)

compute_accuracy(w, m)
#################################################################
